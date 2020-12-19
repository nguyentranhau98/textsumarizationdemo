def insert_cls_sep(raw_text):
    input_text = raw_text.replace('...', '.')
    return raw_text.replace('.', '. [CLS] [SEP] ')

def make_ext_input_file(input_text):
    processed_text = insert_cls_sep(input_text)
    final_input = processed_text.replace('\n', ' ')
    path = '/home/nguyentranhau/Desktop/Final/textsumbert/input.txt'
    with open(path, 'w') as f:
        f.write(final_input)
    return path

def make_abs_input_file(input_text):
    final_input = input_text.replace('\n', ' ')
    print(final_input)
    path = '/home/nguyentranhau/Desktop/Final/textsumbert/input.txt'
    with open(path, 'w') as f:
        f.write(final_input)
    return path

def clean_output(output_text):
    output_text = output_text.replace('.<q>', '. ')
    return output_text.replace('<q>', '. ')

