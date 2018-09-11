TAG_NAME=$1

echo "Trying to delete existing local tag $1 .."
git tag -d $TAG_NAME

echo "Trying to delete existing remote tag $1 .."
git push --delete origin $TAG_NAME

echo "Creating tag for latest commit in branch .."
git tag $TAG_NAME

echo "Pushing tag to origin .."
git push --tags
