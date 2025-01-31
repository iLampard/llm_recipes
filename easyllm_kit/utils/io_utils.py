import dictdatabase as DDB
from easyllm_kit.utils.log_utils import get_logger

logger = get_logger('easyllm_kit')


def initialize_database(output_db: str):
    """
    Initialize the database if it doesn't exist and return the database object.
    """
    db = DDB.at(output_db).read()
    if db is None:
        DDB.at(output_db).create()
        db = {}
        logger.info(f"Initialized new database: {output_db}")
    else:
        logger.info(f"Loaded existing database: {output_db} with {len(db)} entries.")
    return db


def write_to_database(output_db: str, record_idx: str, results: dict, verbose: bool=True):
    """
    Write the results to the database for a given question ID.
    """
    with DDB.at(output_db).session() as (sess, obj):
        obj[record_idx] = results
        sess.write()
        if verbose:
            logger.info(f"Stored answer for record_idx {record_idx}.")
