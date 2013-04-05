#! /usr/bin/python

from __future__ import print_function
import smtplib

username = 'elliot.marsden'
password = raw_input('Password: ')
recipient = raw_input('To: ')
subject = raw_input('Subject: ')
body = raw_input('Body: ')

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
