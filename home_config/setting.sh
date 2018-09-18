#!  /usr/bin/bash
#----------------------------------------------------------
#   Name: setting.sh
#   Author: xyy15926
#   Created at: 2018-08-18 18:29:34
#   Updated at: 2018-08-18 19:04:57
#   Description: 
#----------------------------------------------------------

export HOME_CNF=~/Code/proxy/home_config
ln -sf "$HOME_CNF/vim." ~/.vim
ln -sf "$HOME_CNF/ctags." ~/.ctags
ln -sf "$HOME_CNF/config." ~/.config

if [ -f ~/.profile ]; then
	profile=~/.profile
elif [ -f ~/.bash_profile ]; then
	profile=~/.bash_profile
fi
echo "" >> $profile
echo "# launch personal setting" >> $profile
echo "if [-f $HOME_CNF/bash_profile_addon]; then" >> $profile
echo -e "\t. $HOME_CNF/bash_profile_addon" >> $profile
echo "fi" >> $profile

if [ -f ~/.bashrc ]; then
	bashrc=~/.bashrc
fi
echo "" >> $bashrc
echo "# launch personal setting" >> $bashrc
echo "if [ -f $HOME_CNF/bash_addon ]; then" >> $bashrc
echo -e "\t. $HOME_CNF/bash_addon" >> $bashrc
echo "fi" >> $bashrc

