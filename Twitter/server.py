from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        parsed_url = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # Extract relevant information from query parameters (e.g., oauth_token, oauth_verifier)
        oauth_token = query_params.get('oauth_token', [''])[0]
        oauth_verifier = query_params.get('oauth_verifier', [''])[0]

        # Print or process the received tokens
        print("Received OAuth Token:", oauth_token)
        print("Received OAuth Verifier:", oauth_verifier)

        # You would typically use these tokens to complete the OAuth flow 
        # and obtain access tokens for further API calls

        self.wfile.write(b"Authentication successful! You can close this window now.")

def run_server(server_class=HTTPServer, handler_class=CallbackHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()