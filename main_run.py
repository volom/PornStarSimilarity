from extract_embedding import *
from tqdm import tqdm
from scipy.spatial.distance import cosine
import cv2
from matplotlib import pyplot as plt
import sys
import os

def run():
    input_image = sys.argv[1]
    input_embedding = get_embeddings([input_image])

    df_db = pd.read_csv('photos_db.csv', sep=';')
    df_db['input_cosine'] = ''


    for index in tqdm(range(0, len(df_db)), position=0, leave=True):
        ps_embedding = np.array(list(map(lambda x: float(x), df_db['embedding'][index].replace('[', '').replace(']', '').split(', ')))).reshape(1, 2048)
        df_db['input_cosine'][index] = cosine(input_embedding, ps_embedding)

    df_db.sort_values(by='input_cosine', ascending=True, inplace=True)
    df_db.reset_index(drop=True, inplace=True)

    # Plot photos
    im1 = df_db['filename'][0]
    im2 = df_db['filename'][1]
    im3 = df_db['filename'][2]

    image_input = cv2.imread(input_image)

    Image1 = cv2.imread(f'{os.getcwd()}/ps_photos/{im1}')
    Image2 = cv2.imread(f'{os.getcwd()}/ps_photos/{im2}')
    Image3 = cv2.imread(f'{os.getcwd()}/ps_photos/{im3}')

    # create figure
    fig = plt.figure(figsize=(15, 10))
    
    # setting values to rows and column variables
    rows = 2
    columns = 2
    fig.add_subplot(rows, columns, 1)
    fig.add_subplot(rows, columns, 1)
    
    # showing image
    plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Input image")
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    
    # showing image
    plt.imshow(cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    title1 = str(df_db['ps_name'][0]) + f' | cosine - '+str(df_db['input_cosine'][0])
    plt.title(title1)
    
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    
    # showing image
    plt.imshow(cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    title2 = str(df_db['ps_name'][1]) + f' | cosine - '+str(df_db['input_cosine'][1])
    plt.title(title2)
    
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)
    
    # showing image
    plt.imshow(cv2.cvtColor(Image3, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    title3 = str(df_db['ps_name'][2]) + f' | cosine - '+str(df_db['input_cosine'][2])
    plt.title(title3)
    plt.savefig(f'{os.path.splitext(os.path.basename(input_image))[0]}_RESULT.png')
    plt.show()

if __name__ == "__main__":
    run()