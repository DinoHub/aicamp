from PIL import Image
import numpy as np 
import pandas as pd
import os
import json
import time
import wget
import tarfile
from pprint import pprint
from urllib.parse import urljoin

## participants will type the following into the file with test_model() where they take np array input and give list of string predictions 
"""from eval import eval

    # and call

    eval(test_model, submission_type, team_secret)

"""

 ## Function for submission
def submit_result(submission):
    import requests
    print('Submitting predictions')
    headers = {'content-type' : 'application/json'}
    # url = "https://eizfxz1sfh.execute-api.us-west-2.amazonaws.com/default/competitionScore"
    url = "https://yfpki7bqa9.execute-api.us-east-1.amazonaws.com/default/submit"
    res = requests.post(url, data=json.dumps(submission), headers=headers)
    return res.json()


# def eval(test_model, url, submission_type, team_secret):
def eval_submit(test_model, submission_type, team_secret):
    start = time.time()

    baseurl = 'https://ai-camp.s3-us-west-2.amazonaws.com'
    tar_ext = '.tar.xz'
    tar_fn = submission_type + tar_ext
    url = urljoin(baseurl, tar_fn)

    derek_folder = '/tmp/derek_'+str(submission_type)
    if not os.path.exists(derek_folder):
        print('\nBeginning file download of test images')
        wget.download(url, derek_folder+'.tar')
        try:
            tar = tarfile.open(derek_folder+'.tar', "r:xz")
        except:
            try:
                tar = tarfile.open(derek_folder+'.tar', "r:gz")
            except:
                tar = tarfile.open(derek_folder+'.tar', "r:")
        tar.extractall(path=derek_folder)
        tar.close()
        end = time.time()
        print('\nTime taken for download: {:.3f}s'.format(end-start))
    evan = [] #list of np arrays
    # derek_folder = derek_folder + '/' + str(submission_type)
    test_imgs = os.listdir(derek_folder)
    for img in test_imgs:
        eugene = Image.open(derek_folder+'/'+img) # eugene is opened pillow thingy (bitmap?)
        evan.append(np.array(eugene))
    
    print('\nPredicting...')    
    predictions = test_model(evan) #calls student's test_model() function, feeding them 
                            # the image and getting a prediction
                            # students have to prepare their own py file with the function test_model()
    for bb in predictions:
        bb = bb.lower()

    # print(predictions) #muahaha

    # Preparing results into dataframe
    results=pd.DataFrame({"filename":test_imgs,
                          "prediction":predictions})

    # Output a CSV (optional)
    # results.to_csv('results.csv', index = None)

    # Peparing submission payload
    submission = {
        "team_id": team_secret,
        "submission_type": submission_type
    }
    submission['predictions'] = json.loads(results.to_json(orient='records'))



    end = time.time()
    print('\nTotal time taken for model evaluation: {:.3f}s\n'.format(end-start))
    ## Calling the function to submit
    pprint(submit_result(submission))
    ## DON'T CHEAT >:(
    os.remove(derek_folder+'.tar')
    os.rmdir(derek_folder)

def test():
    print('Hello World \n-Alpheus & Ivan')

# if __name__ == '__main__':
#     eval(test, 'leader_board', 'evan')