# src.data_extractor.py
# convenient, consistent way to run extract sql scripts from CLI
import confuse
import os
import re
import sqlalchemy
import time
import typer
import pandas as pd
from typing import Optional

DIR_DATA_EXTRACT = ['data', 'raw', 'extracted']
app = typer.Typer()


def _create_engine(option='config_group'):
    """Create SQL engine"""
    cfg = confuse.Configuration('data_extractor', __name__)

    cfg.set_file(
        os.path.join('dev', 'config', 'database.yml')
    )  # get private config first
    cfg_private = cfg.get()
    cfg.set_file(os.path.join('config', 'database.yml'))  # then get open config

    # merge in (or at least, the non repo-ed usr/pwd creds)
    for k, v in cfg.get().items():
        v.update(cfg_private[k])
    # TODO (P3) make recursive
    # for k, v in cfg.get().items():
    #     for k1, v1 in v.items():
    #         v1.update(cfg_private[k][k1])

    try:
        kind = cfg[option]['kind'].get(str)
        host = cfg[option]['host'].get(str)
        port = cfg[option]['port'].get(int)
        database = cfg[option]['database'].get(str)
        schema = cfg[option]['schema'].get(str)
        usr = cfg[option]['usr'].get(str)
        pwd = cfg[option]['pwd'].get(str)
    except confuse.ConfigError as e:
        raise e

    try:
        domain = cfg[option]['domain'].get(str)
    except confuse.ConfigError as e:
        domain = None

    if kind == 'mssqlserver':
        if domain is not None:
            # For Windows Auth via domain, we must use FreeTDS (http://www.pymssql.org)
            eng = sqlalchemy.create_engine(
                f'mssql+pymssql://{domain}\{usr}:{pwd}@{host}:{port}/{database}'
            )
        else:
            # Use ODBC, need [driver](https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/
            #                            installing-the-microsoft-odbc-driver-for-sql-server)
            eng = sqlalchemy.create_engine(
                (
                    f'mssql+pyodbc://{usr}:{pwd}@{host}:{port}/{database}'
                    + '?driver=ODBC+Driver+18+for+SQL+Server'
                    + '&trusted_connection=yes'
                ),
                fast_executemany=True,
            )
    elif kind == 'postgres':
        eng = sqlalchemy.create_engine(
            f'postgresql://{usr}:{pwd}@{host}:{port}/{schema}'
        )
    else:
        raise AttributeError('Config file: kind in {mssqlserver, postgres}')

    return eng


@app.command()
def extract_datasets(
    extract_asat_date: Optional[str] = typer.Argument(
        time.strftime("%Y-%m-%d"), help="ISO 27001 e.g. 2021-12-31"
    ),
    csv: Optional[bool] = False,
    dry_run: Optional[bool] = False,
):
    """Execute sql/extract_*.sql scripts against the DB"""

    extract_asat_date_replacement = f"cast('{extract_asat_date}' as TIMESTAMP)"
    typer.echo(f'Extracting datasets as-at date {extract_asat_date}')

    fquery = {'policy_and_claims': 'extract_policy_and_claims.sql'}

    engine = _create_engine()
    cnx = engine.connect()
    rx_sel = re.compile(r'select', re.I)

    for k, v in fquery.items():
        with open(os.path.join('sql', v), 'r') as f:
            typer.echo(f'Extracting {k} using sql/{v}')

            q = f.read()
            # q = q.replace('CURRENT_DATE', extract_asat_date_replacement)
            if dry_run:
                q = rx_sel.sub('select top 10', q)

            res = cnx.execute(sqlalchemy.text(q))
            df = pd.DataFrame.from_records(
                res.all(), coerce_float=False, columns=res.keys()
            )
            res.close()

            fqn = os.path.join(*DIR_DATA_EXTRACT, f'{k}_{extract_asat_date}')
            df.to_parquet(f'{fqn}.parquet', engine='pyarrow')
            typer.echo(f'Saved {fqn}.parquet')

            if csv:
                df.to_csv(f'{fqn}.csv', index_label='rowid')
                typer.echo(f'Also saved all rows to {fqn}.csv')
            else:
                df.iloc[:10].to_csv(f'{fqn}.csv', index_label='rowid')
                typer.echo(f'Saved top 10 rows to {fqn}.csv')

    engine.dispose()


if __name__ == '__main__':
    app()
