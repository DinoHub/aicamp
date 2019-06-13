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
import shutil
### hello curious participant
### say hi to Ivan (I typed this code)


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
def eval_submit(test_model, submission_type, team_secret, batch_size=400):
    start = time.time()
    assert batch_size >= 1, 'Batch size must be >= 1'
    # baseurl = 'https://ai-camp.s3-us-west-2.amazonaws.com' # for everyone
    baseurl = 'https://ai-camp-internal.s3.amazonaws.com' # ONLY FOR TESTERS DO NOT RELEASE
    tar_ext = '.tar.xz'
    tar_fn = submission_type + tar_ext
    url = urljoin(baseurl, tar_fn)

    derek_folder = '/tmp/derek_'+str(submission_type)
    if os.path.exists(derek_folder):
        # os.remove(derek_folder)
        shutil.rmtree(derek_folder)
    if os.path.exists(derek_folder+'.tar'):
        os.remove(derek_folder+'.tar')
 
    # if not os.path.exists(derek_folder):
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

# derek_folder = derek_folder + '/' + str(submission_type)
    test_imgs_all_unsorted = os.listdir(derek_folder)
    test_imgs_all = []
    # push a file, if any, named BINGO.png to the front of the list test_imgs_all
    for alpheus in test_imgs_all_unsorted:
        if alpheus=='BINGO.png':
            test_imgs_all.insert(0,alpheus)
        else:
            test_imgs_all.append(alpheus)

    predictions_all = []
    final_test_imgs_all = []
    num_batches= (len(test_imgs_all)//batch_size)+1
    for batch_idx, b in enumerate(range(0,len(test_imgs_all),batch_size)):
        test_imgs = test_imgs_all[b:min(b+batch_size,len(test_imgs_all))]

        found_bingo = False
        final_test_imgs = []
        evan = [] #list of np arrays
        for img in test_imgs:
            eugene = Image.open(derek_folder+'/'+img) # eugene is opened pillow thingy (bitmap?)
            if img == "BINGO.png":
                found_bingo = True
                evan.insert(0, np.array(eugene))
            else:
                evan.append(np.array(eugene))
                final_test_imgs.append( img )
        
        print('\nPredicting batch {}/{}...'.format(batch_idx+1, num_batches))    
        predictions = test_model(evan) #calls student's test_model() function, feeding them 
                                # the image and getting a prediction
                                # students have to prepare their own py file with the function test_model()
        if found_bingo:
            del predictions[0]

        assert len(predictions) == len(final_test_imgs),'Mismatch in length of predictions({}) vs length of names({})'.format(len(predictions, len(final_test_imgs)))

        for p in range(len(predictions)):
            predictions_all.append(predictions[p])
            final_test_imgs_all.append(final_test_imgs[p])


    

    # Preparing results into dataframe
    results=pd.DataFrame({"filename":final_test_imgs_all,
                          "prediction":predictions_all})

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
    if os.path.exists(derek_folder+'.tar'):
        os.remove(derek_folder+'.tar')
    if os.path.isdir(derek_folder):
        shutil.rmtree(derek_folder, ignore_errors=True)

def test():
    print('Hello World \n-Alpheus & Ivan say hi')

# if __name__ == '__main__':
#     eval(test, 'leader_board', 'evan')