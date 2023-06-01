ssh_address=git@github.com:Ri-chard-Wu/PRLC-final-project-v2.git

echo "*.tree" > .gitignore

git init
git add *
git commit -m "e"
git remote add origin $ssh_address
git push -u origin master