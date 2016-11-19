---
layout: post
title: "Markdown Tutorial"
categories: general
author: scott linderman
excerpt_separator: <!--more-->
comments: true
---

Use this template as a starting point for your post.
The file name is important, it must be of the form
YYYY-MM-DD-shortname.md in order for it to be properly
sorted by date. Don't use underscores in the shortname.

<!--more-->

Use the excerpt_separator to mark the end of the short intro
(that's all that show's up on the homepage)

# This is a first level heading

## This is a second level heading

## This is a third level heading

> This is a
> blockquote
> with
>
> two paragraphs

This is a list:
* A
* B
* C

If your list items span multiple paragraphs, intend the items with three spaces.
Here the list is enumerated.

1.   This is the first sentence.

     This is the second sentence.

2.   And so on...

3.   And so forth...

This is **bold text**.
This is _italic text_.
This is ~~strikeout text~~.

This is an inline equation: $$a^2 + b^2 = c^2$$. You have to use two
dollar signs to open and close, not one like in Latex.
To have a centered equation, write it as a pararaph that starts and
ends with the two dollar signs:

$$
p(\theta \, | \, y) \propto p(\theta) \, 
p(y \, | \, \theta).
$$

I don't think you can do align blocks yet.

This is `inline  code`. 
Code blocks are intended paragraphs with four spaces:

```python
F = lambda n: ((1+np.sqrt(5))**n - (1-np.sqrt(5))**n) / (2**n * np.sqrt(5))
```
This is a figure. Note that `site.base_url` refers to the homepage.
In this case, `abc.png` is located in the `img` folder under root.

![ABC]({{site.base_url}}/img/abc.png)

### References
I've just been copying and pastying references as follows: 

[1] Meeds, Edward, Robert Leenders, and Max Welling. "Hamiltonian ABC." _arXiv preprint arXiv:1503.01916_ (2015). [link](http://arxiv.org/pdf/1503.01916)
...

### Footnotes
Here's my trick for footnotes. You can write HTML inside markdown, so I just create a
div with the id footnotes and then add a link[<sup>1</sup>](#footnotes)

<div id="footnotes"></div>
1. like this.
2. ...

