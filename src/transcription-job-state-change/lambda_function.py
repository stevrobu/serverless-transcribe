import boto3
import json
import os
import re

s3_resource = boto3.resource('s3')

s3 = boto3.client('s3')
ses = boto3.client('ses')
transcribe = boto3.client('transcribe')
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

media_bucket_name = os.environ['MEDIA_BUCKET_NAME']


def s3_key_from_url(s3_url):
    exp = f".*{media_bucket_name}/(.*)"
    match = re.search(exp, s3_url)
    return match.group(1)


def get_s3_metadata(s3_url):
    bucket = media_bucket_name
    key = s3_key_from_url(s3_url)

    return s3.head_object(Bucket=bucket, Key=key)['Metadata']


def get_transcription_job(transcription_job_name):
    job = transcribe.get_transcription_job(
        TranscriptionJobName=transcription_job_name
    )['TranscriptionJob']

    return job


# For a given transcription job, gets the S3 object holding the completed
# transcription data and returns the parsed JSON from that file.
def get_transcript_data(transcription_job):
    transcript_file_uri = transcription_job['Transcript']['TranscriptFileUri']
    transcript_path = transcript_file_uri.split("amazonaws.com/", 1)[1]

    transcript_bucket = transcript_path.split('/', 1)[0]
    transcript_key = transcript_path.split('/', 1)[1]

    s3_object = s3_resource.Object(transcript_bucket, transcript_key).get()
    transcript_json = s3_object['Body'].read().decode('utf-8')
    transcript_data = json.loads(transcript_json)

    return transcript_data


def parse_transcript_data(transcript_data):
    try:
        # Segments are a start and end time associated with a speaker
        segments = transcript_data['results']['speaker_labels']['segments']

        # Generate a list of end times from all the segments
        segment_boundaries = []
        for segment in segments:
            segment_boundaries.append(float(segment['end_time']))
            segment['_items'] = []

        # Loop through all the items and add them to the segment they belong
        # to. Some items don't have a time (like punctuation), and always
        # considered to be part of the same segment as the last seen
        # timestamped item.
        for item in transcript_data['results']['items']:
            active_segment_index = len(segments) - len(segment_boundaries)

            if 'start_time' in item:
                # Check to see which segment we're in
                if float(item['start_time']) > segment_boundaries[0]:
                    # We've moved into the next segment
                    # Add this item to the next segment
                    segments[active_segment_index + 1]['_items'].append(item)
                    # Remove the current segment from the list
                    del segment_boundaries[0]
                else:
                    # Still in the same segment
                    segments[active_segment_index]['_items'].append(item)
            else:
                # Continue with active speaker and segment
                segments[active_segment_index]['_items'].append(item)

        # Each segment is treated as a continuous block of text (i.e., a
        # paragraph). The segments speaker label (e.g., spk_0) is prepended to
        # the text.
        lines = []

        for segment in segments:
            contents = []

            for item in segment['_items']:
                if item['type'] == 'pronunciation':
                    contents.append(' ')

                contents.append(item['alternatives'][0]['content'])

            lines.append(f"{segment['speaker_label']}: {''.join(contents)}")

        text = '\n\n'.join(lines)

        return text
    except Exception:
        return ''


def send_email(to, subject, body):
    ses.send_email(
        Source=os.environ['NOTIFICATION_SOURCE_EMAIL_ADDRESS'],
        Destination={
            'ToAddresses': [
                to
            ]
        },
        Message={
            'Subject': {
                'Data': subject,
                'Charset': 'UTF-8'
            },
            'Body': {
                'Text': {
                    'Data': body,
                    'Charset': 'UTF-8'
                }
            }
        }
    )

def split_string_into_chunks(input_string, chunk_size=50000):

    # Splits an input string into chunks of specified size.

    # Args:
    #    input_string (str): The original string to be split.
    #    chunk_size (int): The desired size of each chunk (default is 50,000 bytes).

    #Returns:
    #    list: A list of substrings, each with a maximum length of chunk_size.

    return [input_string[i:i + chunk_size] for i in range(0, len(input_string), chunk_size)]



def lambda_handler(event, context):
    transcription_job_name = event['detail']['TranscriptionJobName']
    transcription_job_status = event['detail']['TranscriptionJobStatus']

    # Get details about the Amazon Transcribe transcription job that triggered
    # the EventBridge event.
    transcription_job = get_transcription_job(transcription_job_name)

    # Get the S3 URL of the original media file that was sent for transcription
    media_uri = transcription_job['Media']['MediaFileUri']

    # Only proceed if the media file is in the media bucket that belongs to
    # this stack.
    if re.search(media_bucket_name, media_uri).group() is None:
        print('Unknown transcription job')
        return

    notification_email = get_s3_metadata(media_uri)['email']

    if transcription_job_status == 'FAILED':
        # If the job has failed send an email with an explanation of the
        # failure
        print(f"Transcription job failed: {transcription_job_name}")
        print(f"Reason: {transcription_job['FailureReason']}")

        print(f"Sending notification to {notification_email}")
        to = notification_email
        subject = f"Transcription failed for {media_uri}"
        body = f"Reason: {transcription_job['FailureReason']}"
        send_email(to, subject, body)

    elif transcription_job_status == 'COMPLETED':
        # If the job is complete, get the transcript and send it in an email
        print(f"Transcription job complete: {transcription_job_name}")

        transcript_data = get_transcript_data(transcription_job)
        transcripts = transcript_data['results']['transcripts']
        transcript = transcripts[0]['transcript']

        parsed_transcript = parse_transcript_data(transcript_data)

        print(len(parsed_transcript))


        # We need to chunk the transcript to 50K bytes, to keep under the prompt limits for this Claud model.

        chunks = split_string_into_chunks(parsed_transcript)
        overall_summary = ""

        for chunk in chunks:

            llm_start = "Human: Here is a recording of a meeting. Start of Transcript:"  + chunk

            llm_end = """End of Transcript.
            Provide a chronology of the meeting, with time windows and topics. For each topic, provide a summary from the perspective of the meeting organizer.
            If possible, identify speakers by name.
            Then list any action items and next steps.
            Assistant:"""

            concatenation = llm_start + llm_end

            print("Processing 50K chunk")
            print(len(concatenation))

            prompt = json.dumps({
                    "prompt": concatenation,
                    "max_tokens_to_sample": 4096,
                    "top_p": 0.999,
                    "top_k": 250,
                    "temperature": 0.8:
                    })

            model = "anthropic.claude-instant-v1"
            #model = "anthropic.claude-v1"

            response = bedrock.invoke_model(body=prompt, modelId=model)
            print(response)
            response_body = json.loads(response.get("body").read())

            overall_summary += response_body.get("completion") + """
            \n ========================== End of Segment (~ 35 minutes) ========================== \n
            """

        body = ''.join([
            f"Original file: {s3_key_from_url(media_uri)}",
            '\n',
            'Below is a summary of your recorded Chime meeting.',
            ' Note: If your meetig is longet than (approx) 45 minutes, the output is processed in chunks'
            '\n========================== First Meeting Segment ( ~35 minutes) ========================== \n',
            overall_summary,
            '\n==========================\n',
            'Below is the transcript of your recorded Chime meeting. \n\n',
            parsed_transcript
#            chunk
        ])

        print(f"Sending notification to {notification_email}")
        send_email(notification_email, 'Transcript is complete', body)

    else:
        # TODO
        pass
