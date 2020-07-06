import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate

ACCOUNT = "reicouohkubo@gmail.com"
ALIAS = "reicou@i.softbank.jp"
PASSWORD = "aiiquhzjljfrbtcn"

def send_message(subject="def_sub", msg="def_msg"):
    msg = MIMEText("Body: " + msg)
    msg["Subject"] = subject
    msg["From"] = ACCOUNT
    msg["To"] = ALIAS
    msg["Date"] = formatdate(localtime=True)
    s = smtplib.SMTP("smtp.gmail.com", 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(ACCOUNT, PASSWORD)

    s.sendmail(ACCOUNT, ALIAS, msg.as_string())
    s.close()
