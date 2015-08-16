# Remove whatever is in the distribution directory
trash dist/*
# Build wheel distribution
./setup.py bdist_wheel
# Build source distributions
./setup.py sdist
# Sign each distribution
for d in dist/*; do
    gpg --detach-sign -a "$d"
done
# Upload to PyPI
twine upload dist/*
