logfile_name="log1.log"
logfile_path=r'D:\python-workplace\py-learn\day19\logg'


standard_format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
simple_format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
LOGGING_DIC={
    'version':1,
    'disable_existing_loggers':False,
    'formatters':{
        'standard':{
            'format':standard_format
        },
        'simple':{
            'format':simple_format
        }
    },
    'filters':{},
    'handlers':{},
    'loggers':{},
}