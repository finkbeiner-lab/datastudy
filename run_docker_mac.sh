docker run -itd --mount type=bind,source="$HOME/datastudy/bin",target=/app --mount type=bind,source=/Volumes/Finkbeiner-Robodata,target=/gladstone/finkbeiner/robodata --mount type=bind,source=/Volumes/Finkbeiner-Linsley,target=/gladstone/finkbeiner/linsley --mount type=bind,source=/Volumes/Finkbeiner-Barbe,target=/gladstone/finkbeiner/barbe --mount type=bind,source=/Volumes/Finkbeiner-Elia,target=/gladstone/finkbeiner/elia jdlamstein/datastudy