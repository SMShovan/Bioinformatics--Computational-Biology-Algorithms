import skbio  # type: ignore

with open("jupyter_out.txt", "w+") as concatenated_file:
    concatenated_file.write("Hello world" + skbio.title + skbio.art)
