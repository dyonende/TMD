'''
@authors: Dyon van der Ende, Eva den Uijl, Myrthe Buckens
download pdf files of articles
'''
import pandas as pd
import requests
import argparse
import sys
import os

def check_path(path):
    if os.path.isfile(path) == False and os.path.isdir(path) == False:
        print(f"{path} is not valid")
        sys.exit()
        
    if os.path.isdir(path) and path[:-1] != '/':
        path += '/'
    
    return path

def collect_links(input_file):
    '''
    collect the id and url to article for each entry in the data
    '''
    with open(input_file) as infile:
        df = pd.read_csv(infile, delimiter=',')
        df = df[['_id', 'unpaywall_response.free_fulltext_url']]
        df = df.drop_duplicates()
        return df   
        
def download_pdf(ids_and_links, output_path):
    '''
    try to download the pdf at the specified url and saved 
    it to a file with the id as name
    '''
    ids = list(ids_and_links['_id'])
    links = list(ids_and_links['unpaywall_response.free_fulltext_url'])
    number_of_links = len(ids)
    
    for i in range(number_of_links):
        print(str(i/number_of_links*100)[:4]+"%", end="\r") #print progress
        id = ids[i]
        link = links[i]
        filename = id
        try:            
            r = requests.get(link, stream=True)
            if r.headers['content-type'].find("pdf") > -1:
                filename += ".pdf"
            elif r.headers['content-type'].find("html") > -1:
                filename += ".html"
                
            with open(output_path+filename, 'wb') as outfile:
                outfile.write(r.content)
                
        except KeyboardInterrupt:
            sys.exit()
        except:
            print(id +" failed to download")
        
        
          

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='file path to data')
    parser.add_argument('output_path',
                        help='file path to store pdf\'s')
    args = parser.parse_args()
    input_file = args.input_file
    output_path = args.output_path
    
    check_path(input_file)
    output_path = check_path(output_path)

    ids_and_links = collect_links(input_file)
    download_pdf(ids_and_links, output_path)

if __name__ == '__main__':
    main()
