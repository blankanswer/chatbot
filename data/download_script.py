from datasets import load_dataset, Dataset, load_from_disk
import os


def main():
    os.makedirs('./promptbook', exist_ok=True)
    promptbook = load_dataset('NYUSHPRP/ModelCofferPromptBook', split='train')
    print(promptbook)
    promptbook.save_to_disk('./promptbook')
    #
    # roster = load_dataset('NYUSHPRP/ModelCofferRoster', split='train')
    # roster.save_to_disk('./roster')


def load():
    roster = load_from_disk('./roster')
    print(roster)


def test():
    promptbook = load_from_disk('./promptbook')
    print(promptbook[0]['image'])


# def drop_metadata_checked_column():
#     ModelCofferMetadata = load_dataset('NYUSHPRP/ModelCofferMetadata', split='train')
#     ModelCofferMetadata = ModelCofferMetadata.remove_columns(['checked'])
#     ModelCofferMetadata.push_to_hub('NYUSHPRP/ModelCofferMetadata', split='train')


if __name__ == '__main__':
    main()
