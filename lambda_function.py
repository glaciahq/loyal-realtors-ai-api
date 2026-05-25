import json
import boto3
from prompt import SYSTEM_PROMPT

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,x-api-key",
    "Content-Type": "application/json"
}

def lambda_handler(event, context):

    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({})
        }

    try:

        body = json.loads(event.get("body", "{}"))

        user_message = body.get("message", "")

        request_body = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "max_tokens": 300,
            "temperature": 0.5
        }

        response = bedrock.invoke_model(
            modelId="qwen.qwen3-32b-v1:0",
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(
            response["body"].read()
        )

        print("BEDROCK RESPONSE:", response_body)

        # safer parsing
        reply = ""

        if "output" in response_body:
            reply = response_body["output"].get("text", "")

        elif "choices" in response_body:
            reply = (
                response_body["choices"][0]
                .get("message", {})
                .get("content", "")
            )

        elif "outputs" in response_body:
            reply = (
                response_body["outputs"][0]
                .get("text", "")
            )

        else:
            reply = str(response_body)

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({
                "reply": reply
            })
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({
                "error": str(e)
            })
        }
