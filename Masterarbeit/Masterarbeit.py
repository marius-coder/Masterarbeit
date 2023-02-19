# -*- coding: cp1252 -*-
import requests

params = {
  'access_key': 'be822d72138ce17b08d3ff4dfe912f63',
  'query': 'Vienna'
}

api_result = requests.get('http://api.weatherstack.com/current', params)

api_response = api_result.json()

print('Current temperature in %s is %d°C' % (api_response['location']['name'], api_response['current']['temperature']))

print("")