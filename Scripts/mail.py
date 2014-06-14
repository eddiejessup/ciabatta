#! /usr/bin/python


import smtplib

username = 'elliot.marsden'
password = input('Password: ')
recipient = input('To: ')
subject = input('Subject: ')
body = input('Body: ')

headers = '\r\n'.join([
    'from: %s' % username,
    'subject: %s' % subject,
    'to: %s' % recipient,
    'mime-version: 1.0',
    'content-type: text/html'])

session = smtplib.SMTP('smtp.gmail.com', 587)
session.ehlo()
session.starttls()
session.login(username, password)
session.sendmail(username, recipient, headers + '\r\n\r\n' + body)
