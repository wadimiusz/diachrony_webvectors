#!/usr/bin/env python
# coding: utf-8

import requests
import json


def tag_ud(port, text='Do not forget to pass some text as a string!'):
    # UDPipe tagging for any language you have a model for.
    # Demands UDPipe REST server (https://ufal.mff.cuni.cz/udpipe/users-manual#udpipe_server)
    # running on a port defined in webvectors.cfg
    # Start the server with something like:
    # udpipe_server --daemon 66666 MyModel MyModel /opt/my.model UD

    # Sending user query to the server:
    ud_reply = requests.post('http://localhost:%s/process' % port,
                             data={'tokenizer': '', 'tagger': '', 'data': text}).content

    # Getting the result in the CONLLU format:
    processed = json.loads(ud_reply.decode('utf-8'))['result']

    # Skipping technical lines:
    content = [l for l in processed.split('\n') if not l.startswith('#')]

    # Extracting lemmas and tags from the processed queries:
    tagged = [w.split('\t')[2].lower() + '_' + w.split('\t')[3] for w in content if w]
    poses = [t.split('_')[1] for t in tagged]
    return poses

