from flask import Flask, request, jsonify, Response
import requests
import json
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def main():

    try:
        # =========================================
        # INPUT
        # =========================================
        input_value = (
            request.args.get("user")
            or request.args.get("username")
            or request.args.get("id")
        )

        if not input_value:
            return jsonify({
                "success": False,
                "message": "Provide ?user=username_or_id"
            }), 400

        # =========================================
        # CHECK IF ID
        # =========================================
        is_telegram_id = input_value.isdigit()

        telegram_id = input_value

        # =========================================
        # USERNAME => FIRST API
        # =========================================
        if not is_telegram_id:

            first_api_url = (
                ""https://paid.proportalx.workers.dev/tg"
                f"?key=my&username={input_value}"
            )

            first_response = requests.get(
                first_api_url,
                timeout=60,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            try:
                first_data = first_response.json()
            except Exception:
                return jsonify({
                    "success": False,
                    "message": "Invalid response from first API"
                }), 500

            # =========================================
            # EXTRACT ID
            # =========================================
            telegram_id = (
                first_data.get("id")
                or first_data.get("result", {}).get("id")
                or first_data.get("data", {}).get("id")
            )

            if not telegram_id:
                return jsonify({
                    "success": False,
                    "message": "ID not found from first API"
                }), 404

        # =========================================
        # SECOND API
        # =========================================
        second_api_url = (
            "https://openosintx.vippanel.in/tginfo.php"
            f"key=SVZGP&number={telegram_id}"
        )

        second_response = requests.get(
            second_api_url,
            timeout=60,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        second_text = second_response.text

        # =========================================
        # RETURN JSON IF POSSIBLE
        # =========================================
        try:
            second_json = json.loads(second_text)

            return Response(
                json.dumps(second_json, indent=2),
                status=200,
                mimetype="application/json"
            )

        except Exception:

            return Response(
                second_text,
                status=200,
                mimetype="text/plain"
            )

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================
# RENDER PORT FIX
# =========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port
    )
