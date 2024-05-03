import boto3

s3 = boto3.resource('s3',
                     endpoint_url='http://10.8.82.11',
                     aws_access_key_id='rf3-dsm-user-deepsim', 
                     aws_secret_access_key='Sk/AGLR0oFq7IucyBIh8syi0jJi4Kyl6R+v0wDN/'
                    )

# Print out bucket names
for bucket in s3.buckets.all():
    print(Rf"Bucket name = {bucket.name}")

# Create bucket object
s3_bucket = s3.Bucket('rf3-dsm-bkt-deepsim')

# Upload an object to the bucket
print('Starting object upload')
with open(r'\\rf3svap116n2\c$\Users\mfg_kxiong\Documents\LotPredictorAnalysis\Data\LP_ActArrivals_20240412_0252.csv', 'rb') as data:
    s3_bucket.put_object(Key='LP_ActArrivals_20240412_0252.csv', Body=data)
print('Ended object upload')

# Get the list of objects in the bucket using the Resource concept (s3 is a Resource)
# NOTE: Resource is nicer to use than Client. The Client concept was the original AWS API abstraction
for obj in bucket.objects.all():
    print(obj.key, obj.last_modified)

s3_object = s3.Object('rf3-dsm-bkt-deepsim', 'LP_ActArrivals_20240412_0252.csv')
print('Done printing using Resource concept\n')

# Delete an object in a Bucket
print('Deleting object')
s3_object.delete()
print('Done... Deleting object')
