from memorag import MemoRAGLite


def main():
    pipe = MemoRAGLite(
        gen_model_name_or_path="models/memorag-qwen2-7b-inst",
        ret_model_name_or_path="models/bge-m3",
    )

    context = open("examples/harry_potter.txt").read()
    pipe.memorize(context, save_dir="cache/harry_potter", print_stats=True)
    print(pipe("书里密室被打开了几次？"))


if __name__ == "__main__":
    main()
