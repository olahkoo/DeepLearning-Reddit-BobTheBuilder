import praw

reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='xC4cZvfsAnv9RQ', client_secret="xahwuyDtyCOzTay_NyoBOh7-e7M",
                     username='da189a5d30c16', password='123456789')

reddit.read_only = True

subreddit = reddit.subreddit('redditdev')

print(subreddit.display_name)  # Output: redditdev
print(subreddit.title)         # Output: reddit Development
print(subreddit.description) 

print("-----------------------------------------------------------")
for submission in subreddit.hot(limit=1):
    print(submission.title)  # Output: the submission's title
    print(submission.score)  # Output: the submission's score
    print(submission.id)     # Output: the submission's ID
    print(submission.url)

    top_level_comments = list(submission.comments)
    all_comments = submission.comments.list()

    for top_level_comment in top_level_comments:
        print(f"COMMENT: {top_level_comment} {top_level_comment.body} {top_level_comment.score}")
