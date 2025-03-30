# Journal
Willow Cunningham

This document contains notes, thoughts, todos, etc considered in the development of the project. Read at your own risk.

## 3/29/2025

Installing pytorch to get started. On the broadwell machine, need to use apt. made the mistake of `sudo apt update && sudo apt upgrade`-ing, which apparently hasnt been done in a while. To install, will `sudo apt-get install python3-torch`.

Went ahead and installed on raptorlake while waiting. Now to download the examples from [pytorch/examples](https://github.com/pytorch/examples/tree/main)

Okay, so what I want to do actually is characterize individual *layers* and then we can just know what we are dealing with. That easy right?

As a first start, I created a script that can define a net that just has linear layers with a ton of parameters, and times it.
Next, I should make it into an importable function and set up an energy test fixture script
