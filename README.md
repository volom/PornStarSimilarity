# PornStarSimilarity
Search the most similar face porn actress for your input photo from more than 5k available photos. 
Fetching porn stars photos, extraction, and embedding their faces for the next cosine similarity estimation with face embedding of the input photo.

Algorithm and core code taken from https://github.com/volom/Face_Similarity

The collection of photos has large size so take it from Google Drive - https://drive.google.com/drive/folders/1IPA5uzxQup0CD4yh31uwmnAoiTv18khJ?usp=sharing

--------------------------------

🛠 Usage:

1. Copy photos from given Google Drive link: take all photos from folder "pornhub_800best", unzip photos in archive "ps_photos.zip" from folder "pornhub_all_init" and take all of them in folder "ps_photos". Copy to repo database with face vectors - *photos_db.csv*
2. Just run the script "main_run.py":
```
>> pip install -r requirements.txt
>> cd PornStarSimilarity
>> python3 main_run.py YOUR_PHOTO.jpg

```

--------------------------------

☝️☝️☝️ A few remarks

Data was scraped from https://rt.pornhub.com/pornstars and https://rt.pornhub.com/pornstars/top?si=1. Photos taken from https://www.iafd.com/

The python code of the scraping process is in *pornhub_data_collection.ipynd*. It is not used in the running process, just for demonstration purposes of how to collect the database.

You are free to add new photos to your photo database - *main_run.py* extract vectors for the new photos in the directory before analyzing your input image. Extraction vectors from new photos take some time but the information is being saved in the database and then reuse so it will take much less time.
