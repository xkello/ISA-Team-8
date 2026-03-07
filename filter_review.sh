#!/bin/bash

# getting rid of the text attribute
# since it is too long and we may not be able to load it into pandas

cat original_data/yelp_json/yelp_academic_dataset_review.json | jq -c '{
review_id,
user_id,
business_id,
stars,
useful,
funny,
cool,
date
}'

