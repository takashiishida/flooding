

def make_query_from_args(args):
    args = vars(args)
    
    options_list = []
    for (k, v) in args.items():        
        if v == None or v == "" or k == 'gpu_id':
            continue
        if k == 'tags':
            for kw in v.split(','):
                option = "tags.\"{}\" = \"{}\"".format(*kw.split(':'))
        elif k == 'labels':
            for label in v.split(','):
                option = "tags.\"{}\" = \"True\"".format(label)
        else:
            option = "params.{} = '{}'".format(k, v)     

        options_list.append(option)
    options = " and ".join(options_list)   
    return options    