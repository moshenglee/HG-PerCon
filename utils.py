#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : lml
# @File    : utils.py
# @Software: PyCharm
import logging
from sheen import Str, ColoredHandler


class mylogger():
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.handle = ColoredHandler()
        self.handle.setFormatter({
            logging.DEBUG: Str.green('[%(asctime)s] - %(levelname)s | %(message)s'),
            logging.INFO: Str.blue('[%(asctime)s] - %(levelname)s | %(message)s'),
            logging.WARNING: Str.yellow('[%(asctime)s] - %(levelname)s | %(message)s'),
            logging.ERROR: Str.red('[%(asctime)s] - %(levelname)s | %(message)s'),
            logging.CRITICAL: Str.LIGHTRED('[%(asctime)s] - %(levelname)s | %(message)s'),

        })
        self.logger.addHandler(self.handle)

    def debug(self, msg):
        return self.logger.debug(msg)

    def info(self, msg):
        return self.logger.info(msg)

    def warning(self, msg):
        return self.logger.warning(msg)

    def error(self, msg):
        return self.logger.error(msg)

    def critical(self, msg):
        return self.logger.critical(msg)



