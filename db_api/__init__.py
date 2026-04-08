import pymysql
import os


class CustomOperation():
    def __init__(self):
        self.MySQLCommand = None

    def execute(self, cursor: pymysql.cursors.Cursor):
        try:
            cursor.execute(self.MySQLCommand)
            return cursor.fetchall()
        except Exception as e:
            print(e)
            raise e


class CustomTransaction():
    def __init__(self):
        self.is_connecting = False
        self.host     = os.environ.get("CPHOS_DB_HOST", "")
        self.port     = int(os.environ.get("CPHOS_DB_PORT", "3306"))
        self.user     = os.environ.get("CPHOS_DB_USER", "")
        self.db       = os.environ.get("CPHOS_DB_NAME", "")
        self.password = os.environ.get("CPHOS_DB_PASSWORD", "")

        self.conn   = None
        self.cursor = None
        self.connect()

    def connect(self):
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db,
            )
        except Exception as exc:
            raise ConnectionError("Connect to CPHOS database failed.") from exc
        self.cursor = self.conn.cursor()
        self.is_connecting = True

    def executeOperation(self, operation: CustomOperation):
        if not self.is_connecting:
            raise RuntimeError("Not connected to database. Call connect() first.")
        try:
            result = operation.execute(self.cursor)
            return result
        except Exception as e:
            self.conn.rollback()
            self.conn.close()
            self.is_connecting = False
            raise

    def __call__(self, *args, **kwargs):
        return self.executeOperation(*args, **kwargs)

    def commit(self):
        if not self.is_connecting:
            raise RuntimeError("Not connected to database.")
        try:
            self.conn.commit()
            self.conn.close()
            self.is_connecting = False
        except Exception as e:
            raise

    def rollBack(self):
        if not self.is_connecting:
            raise RuntimeError("Not connected to database.")
        try:
            self.conn.rollback()
            self.conn.close()
            self.is_connecting = False
        except Exception as e:
            raise
