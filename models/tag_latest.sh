TAG_NAME=$1

if [ $TAG_NAME -eq 0 ]
  then
    echo "Please supply tagname! In form of [a-zA-Z-0-9]*-[0-9.]*$"
fi

echo "Trying to delete existing local tag $1 .."
git tag -d $TAG_NAME

echo "Trying to delete existing remote tag $1 .."
git push --delete origin $TAG_NAME

echo "Creating tag for latest commit in branch .."
git tag $TAG_NAME

echo "Pushing tag to origin .."
git push --tags
