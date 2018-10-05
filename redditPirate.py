from praw import Reddit
from praw.models import Subreddit, Submission, Comment, MoreComments

import time
import json

# Használt api végpontok a commentekhez
# [/r/subreddit]/api/info
# /api/morechildren

# https://praw.readthedocs.io/en/latest/code_overview/models/comment.html?highlight=Comment
class CommentDTO(object):

    def __init__(self, fullname, author_fullname, body, parent_id, score, is_submitter):
        self.fullname = fullname
        self.author_fullname = author_fullname
        self.body = body
        self.parent_id = parent_id
        self.score = score
        self.is_submitter = is_submitter

    def _asdict(self):
        return self.__dict__

# Felhasználó készítés
# https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps

reddit = Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='xC4cZvfsAnv9RQ',
                     client_secret='xahwuyDtyCOzTay_NyoBOh7-e7M',
                     username='da189a5d30c16',
                     password='123456789')

# Csak olvasni fogunk
reddit.read_only = True

# subreddit = reddit.subreddit('redditdev')
subreddit = reddit.subreddit('AskReddit')

print(subreddit.fullname)
print(subreddit.title)
print(subreddit.display_name_prefixed)
print(subreddit.description)

customSubmissionList = [ reddit.submission(id="9lh5i0") ]

print('-- Submission start here --')
for submission in customSubmissionList:
#for submission in subreddit.hot(limit=1):
    print(submission.fullname)
    print(submission.author_fullname)
    print(submission.title)
    print(submission.selftext)
    print(submission.score)

    # Minden MoreComments kifejtése
    while True:
        try:
            submission.comments.replace_more(limit=None)
            break
        except Exception as e:
            print(f'Handling replace_more exception. {e}')
            time.sleep(1)

    # Ezek a ténylegesen csak legfelső szinten álló kommentek
    # Comment vagy MoreComments példányok
    # top_level_comments = list(submission.comments)
    # for top_level_comment in top_level_comments:
        # print(f'COMMENT: {top_level_comment.body} {top_level_comment.score}')

    serializableComments = []

    # BFS
    comment_queue = submission.comments[:]
    while comment_queue:
        comment = comment_queue.pop(0)

        commentDTO = CommentDTO(
            fullname= comment.fullname,
            author_fullname= comment.author_fullname if hasattr(comment, 'author_fullname') else 'unknown_author',
            body= comment.body,
            parent_id= comment.parent_id,
            score= comment.score,
            is_submitter= comment.is_submitter
        )

        serializableComments.append(commentDTO)

        # print(f'CommentBody: {commentDTO.body}')

        comment_queue.extend(comment.replies)

    with open('data4.json', 'w') as outfile:
        json.dump(serializableComments, outfile, indent=4, sort_keys=True, default=lambda o: o._asdict())

    print(f'Summission comment counter: {submission.num_comments}')
    print(f'Collected  comment counter: {len(submission.comments.list())}')
    print('-- END --')

    # Flattened lista, valszeg nem fog kelleni
    # all_comments = submission.comments.list()

