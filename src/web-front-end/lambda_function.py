# The static webpage used to upload media files for transcriptions uploads the
# files directly to S3. In order for this to work, the POST request made by the
# form on the website to the S3 bucket URL must include several things. These
# include a base 64 encoded copy of the policy used to control uploads to the
# bucket, as well as a signature created using the V4 signing method. There are
# some other things S3 expects in the POST data as well. This function returns
# a JSON object containing those data so the webpage has access to those
# values. It backs an API Gateway endpoint.

import json
import os
import base64
import hashlib
import hmac
from datetime import datetime, timedelta

AMZ_ALGORITHM = 'AWS4-HMAC-SHA256'


# Returns information about the values used to derive the signing key used to
# sign the POST policy for S3 requests
# In the format:
# <your-access-key-id>/<date>/<aws-region>/<aws-service>/aws4_request
# eg: AKIAIOSFODNN7EXAMPLE/20130728/us-east-1/s3/aws4_request
def signing_credentials(signing_time):
    date_stamp = signing_time.strftime('%Y%m%d')
    region = os.environ['AWS_REGION']

    return f"{os.environ['AWS_ACCESS_KEY_ID']}/{date_stamp}/{region}/s3/aws4_request"


# Returns an POST policy used for making authenticated POST requests to S3
# https://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-HTTPPOSTConstructPolicy.html
def s3_post_policy(signing_time, ttl=60):
    expiration_date = datetime.utcnow() + timedelta(minutes=ttl)

    print(f"== EXPIRES == {expiration_date.strftime('%Y-%m-%dT%H:%M:%SZ')}")

    return {
        'expiration': expiration_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'conditions': [
            {'bucket': os.environ['MEDIA_BUCKET']},

            {'x-amz-algorithm': AMZ_ALGORITHM},
            {'x-amz-credential': signing_credentials(signing_time)},
            {'x-amz-date': signing_time.strftime('%Y%m%dT%H%M%SZ')},
            {'x-amz-security-token': os.environ['AWS_SESSION_TOKEN']},

            ["starts-with", "$success_action_redirect", ""],
            ["starts-with", "$x-amz-meta-email", ""],
            ["starts-with", "$x-amz-meta-languagecode", ""],
            ["starts-with", "$x-amz-meta-channelidentification", ""],
            ["starts-with", "$x-amz-meta-maxspeakerlabels", ""],
            ["starts-with", "$key", "audio/"],
            # See link below for why we shouldn't be checking the Content-Type. We are already resticting the file types for the input.
            # https://stackoverflow.com/questions/22073237/allowing-multiple-content-types-in-http-post-amazon-s3-upload-policy-document
            # ["starts-with", "$Content-Type", "video/"],
            # ["starts-with", "$Content-Type", "audio/"]
            ["starts-with", "$Content-Type", ""]
        ]
    }


# Returns a HMAC-SHA256 hash of the given message, using the given key
def digest(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


# Derives a signing key for an AWS V4 signature
# See step 1: https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html
def aws_v4_signing_key(access_key, date_stamp, region, service):
    date_key = digest(('AWS4' + access_key).encode('utf-8'), date_stamp)
    date_region_key = digest(date_key, region)
    date_region_service_key = digest(date_region_key, service)

    signing_key = digest(date_region_service_key, 'aws4_request')

    return signing_key


# Returns an AWS V4 signature for the given string. Generates a signing key
# for S3 in the Lambda execution region
# See step 2: https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html
def aws_v4_signature(signing_time, string_to_sign):
    date_stamp = signing_time.strftime('%Y%m%d')
    region = os.environ['AWS_REGION']
    service = 's3'

    signing_key = aws_v4_signing_key(os.environ['AWS_SECRET_ACCESS_KEY'], date_stamp, region, service)

    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    return signature


def lambda_handler(event, context):
    signing_time = datetime.utcnow()
    print("Generating POST policy and AWS V4 signature.")
    print(f"== SIGNED ===  {signing_time.strftime('%Y-%m-%dT%H:%M:%SZ')}")

    # A POST policy for S3 requests
    post_policy = s3_post_policy(signing_time)

    post_policy_json = json.dumps(post_policy)
    post_policy_json_b64 = base64.b64encode(post_policy_json.encode('utf-8')).decode('utf-8')

    # An AWS V4 signature of the POST policy
    policy_signature = aws_v4_signature(signing_time, post_policy_json_b64)

    api_id = os.environ['API_ID']
    region = os.environ['AWS_REGION']
    redirect_url = f"https://{api_id}.execute-api.{region}.amazonaws.com/transcribe"

    html = open('index.html', 'r', encoding='utf-8').read()
    html = html.replace('__s3_policy__', post_policy_json_b64)
    html = html.replace('__s3_amz_algorithm__', AMZ_ALGORITHM)
    html = html.replace('__s3_amz_signature__', policy_signature)
    html = html.replace('__s3_amz_date__', signing_time.strftime('%Y%m%dT%H%M%SZ'))
    html = html.replace('__s3_amz_credential__', signing_credentials(signing_time))
    html = html.replace('__s3_amz_security_token__', os.environ['AWS_SESSION_TOKEN'])
    html = html.replace('__bucket_domain_name__', os.environ['MEDIA_BUCKET_DOMAIN_NAME'])
    html = html.replace('__s3_success_action_redirect__', redirect_url)

    return {
        'statusCode': 200,
        'headers': {'content-type': 'text/html'},
        'body': html
    }
