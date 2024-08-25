import tweepy

# Twitter API credentials
consumer_key = "3rJOl1ODzm9yZy63FACdg"
consumer_secret = "5jPoQ5kQvMJFDYRNE8bQ4rHuds4xJqhvgNJM4awaE8"

# Authenticate to Twitter
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret)

# Redirect user to Twitter to authorize
try:
    redirect_url = auth.get_authorization_url()
    print(f"Visit this URL to authorize: {redirect_url}")
except tweepy.TweepyException as e:
    print(f"Error! Failed to get request token: {e}")
    exit()

# Get the verifier code from the user
verifier = input("Enter the PIN displayed in your browser after authorization: ")

try:
    auth.get_access_token(verifier)
    print("Access token obtained successfully!")
except tweepy.TweepyException as e:
    print(f"Error! Failed to get access token: {e}")
    exit()

# Create API object with the authenticated user credentials
api = tweepy.API(auth)

# Test the API by fetching your own user information
try:
    user = api.me()
    print(f"Authenticated as: {user.name}")
    print(f"Twitter Handle: {user.screen_name}")
    print(f"Followers: {user.followers_count}")
except tweepy.TweepyException as e:
    print(f"Error! Failed to fetch user details: {e}")
