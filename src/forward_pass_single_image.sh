rm -rf out/*
python3 Denoiser.py -i $1 -o denoised.jpg
python3 WordSegmentator.py -i out/denoised.jpg
python3 main.py --testimg out/0.png --wordbeamsearch