from pathlib import Path

def combine_files(source: [Path], destination: Path, source_encoding: str = "utf-8", destination_encoding: str = "utf-8") -> None:
    with open(destination, "w", encoding=destination_encoding) as outfile:
        for idx, file_path in enumerate(source):
            with open(file_path, "r", encoding=destination_encoding) as infile:
                for line_num, line in enumerate(infile):
                    if idx > 0 and line_num == 0:
                        continue
                    outfile.write(line)