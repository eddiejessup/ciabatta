#! /bin/bash

sudo apt-get update
echo "~~~ Update done!"
sudo apt-get dist-upgrade
echo "~~~ Upgrade done!"
sudo apt-get autoremove
echo "~~~ Autoremove done!"
sudo apt-get autoclean
echo "~~~ Autoclean done!"
echo "~~~ System update finished!"
exit