---
layout: page
title: Schedule
permalink: /schedule/
---

# Spring 2017
<div class="upcoming">
  {% for mtg in site.meetings reversed %}
  {% capture semstart %}{{'2017-01-01 00:00:00 -0500' | date: '%s'}} {% endcapture %}
  {% capture semstop %}{{'2017-06-01 00:00:00 -0500' | date: '%s'}} {% endcapture %}
  {% capture nowunix %}{{'now' | date: '%s'}}{% endcapture %}
  {% capture mtgtime %}{{mtg.date | date: '%s'}}{% endcapture %}
  {% if mtgtime > semstart and mtgtime < semstop %}
  <div class="meeting">
     <p>
	    <b> When: </b> {{ mtg.date | date: "%B %-d, %Y" }}  {{ mtg.time }}
	    <br>
	    <b> Where: </b> {{ mtg.location }}
	    <br>
	    <b> Presenter: </b> {{ mtg.presenter }}
	    <br>
	    <b> Scribe: </b> {{ mtg.scribe }}
	    <br>
	  </p>
	  <div class="excerpt">
            {{ mtg.content }}
          </div>
	  <br>
	 </div>
	 {% endif %}
{% endfor %}
</div>


# Fall 2016

<div class="upcoming">
  {% for mtg in site.meetings reversed %}
  {% capture semstart %}{{'2016-08-30 00:00:00 -0500' | date: '%s'}} {% endcapture %}
  {% capture semstop %}{{'2017-01-01 00:00:00 -0500' | date: '%s'}} {% endcapture %}
  {% capture mtgtime %}{{mtg.date | date: '%s'}}{% endcapture %}
  {% if mtgtime > semstart and mtgtime < semstop %}
  <div class="meeting">
     <p>
	    <b> When:  </b> {{ mtg.date | date: "%B %-d, %Y" }}  {{ mtg.time }}
	    <br>
	    <b> Where: </b> {{ mtg.location }}
	    <br>
	    <b> Presenter: </b> {{ mtg.presenter }}
	    <br>
	    <b> Scribe: </b> {{ mtg.scribe }}
	    <br>
	  </p>
	  <div class="excerpt">
            {{ mtg.content }}
          </div>
	  <br>
	 </div>
	 {% endif %}
   {% endfor %}

</div>
