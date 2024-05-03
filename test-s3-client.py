import boto3

s3_client = boto3.client('s3', 
                         endpoint_url='http://10.8.82.11', 
                         aws_access_key_id='rf3-dsm-user-deepsim', 
                         aws_secret_access_key='Sk/AGLR0oFq7IucyBIh8syi0jJi4Kyl6R+v0wDN/'
                        )

# Get the list of objects using the Client concept
response = s3_client.list_objects_v2(Bucket='rf3-dsm-bkt-deepsim')

for content in response['Contents']:
    obj_dict = s3_client.get_object(Bucket='rf3-dsm-bkt-deepsim', Key=content['Key'])
    print(content['Key'], obj_dict['LastModified'])

print('Done printing using Client concept\n')
