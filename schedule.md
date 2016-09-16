---
layout: page
title: Schedule
permalink: /schedule/
---

## Upcoming Meetings 

<div class="upcoming">
  {% for mtg in site.meetings %}
  {% capture nowunix %}{{'now' | date: '%s'}}{% endcapture %}
  {% capture mtgtime %}{{mtg.date | date: '%s'}}{% endcapture %}
  {% if mtgtime > nowunix %}
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

## Previous Meetings

<div class="upcoming">
  {% for mtg in site.meetings reversed %}
  {% capture nowunix %}{{'now' | date: '%s'}}{% endcapture %}
  {% capture mtgtime %}{{mtg.date | date: '%s'}}{% endcapture %}
  {% if mtgtime < nowunix %}
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
