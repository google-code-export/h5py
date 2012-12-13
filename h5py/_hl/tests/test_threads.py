
"""
    Threading support test module.
"""

import numpy as np

from .common import ut, TestCase

import h5py

import threading


class TestIterateDeadlock(TestCase):

    """
        Checks for deadlocks in calls involving attribute iteration

        See also issue 247 (deadlock with attribute iteration).
    """

    def setUp(self):
        self.fname = self.mktemp()
        with h5py.File(self.fname,'w') as f:
            for i in range(100):
                f.attrs['attr%d'%i] = i

    def read_all_attrs(self):
        """ Support function: returns all attributes of the root group as a dict
        """
        with h5py.File(self.fname,'r') as f:
            return dict(f.attrs)

    def read_only_in_thread(self):
        h5py._errors.silence_errors()  # threading bug where errors are not shut off
        for idx in xrange(100):
            all_attrs = self.read_all_attrs()

    def test_attribute_iteration(self):
        """ Check for deadlocks in attribute listing """
        
        thread = threading.Thread(target=self.read_only_in_thread)
        thread.start()
        for idx in xrange(100):
            self.read_all_attrs()
        thread.join()
