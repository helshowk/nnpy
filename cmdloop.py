#!/usr/bin/env python2

import sys, operator, readline, traceback

class CommandLoop():
    def __init__(self, ctx=dict(), exit_cmd='q', prompt=':'):
        self.exit_cmd = exit_cmd
        self.cmds = dict()
        self.descriptions = dict()
        self.prompt =prompt
        self.ctx = ctx
    
    def start(self, **kwargs):
        cmd = ''
        while (cmd <> self.exit_cmd):
            cmd = raw_input(self.prompt)
            cmd = cmd.split(' ')
            args = cmd[1:]
            cmd = cmd[0]
            if cmd == 'help':
                self.helpCmd()
            elif self.cmds.has_key(cmd):
                #self.cmds[cmd](self.ctx, *args)
                try:
                    self.cmds[cmd](self.ctx, *args)
                except Exception,e:
                    type_, value_, traceback_ = sys.exc_info()
                    print "Traceback (most recent call last): "
                    print ''.join(traceback.format_tb(traceback_))
                    print str(type_) + ": " + str(value_)
    
    def helpCmd(self):
        print "\nAvailable Commands:\n\n"
        for k,v in self.descriptions.items():
            print "\t" + k + " - " + v
    
    def addCmd(self, cmd, f, description=''):
        # f is a function to execute, cmd is the command
        self.cmds[cmd] = f
        self.descriptions[cmd] = description

