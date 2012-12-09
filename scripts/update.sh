#! /bin/bash

apt-get update
echo "~~~ Update done!"
apt-get dist-upgrade
echo "~~~ Upgrade done!"
apt-get autoremove
echo "~~~ Autoremove done!"
apt-get autoclean
echo "~~~ Autoclean done!"
echo "~~~ System update finished!"
