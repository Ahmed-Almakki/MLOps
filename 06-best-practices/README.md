<!--
This section explains the importance of adding an `__init__.py` file in the test directory. 
Including this file tells Python that the directory should be treated as a package, 
which is necessary for proper test discovery and import resolution.
-->
you must add init file in the test to tell python that test is python package, so that test can be discoverd