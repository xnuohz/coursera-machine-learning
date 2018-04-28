# -*- coding:utf-8 -*-


def readFile(filename):
    file = open(filename, 'r')
    content = file.read()
    file.close()
    return content
