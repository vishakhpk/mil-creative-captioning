#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Talk with a model using a web UI.

## Examples

```shell
parlai interactive_web -mf "zoo:tutorial_transformer_generator/model"
```
"""


import torch
bart = torch.hub.load('pytorch/fairseq', 'bart.large')
bart.eval()
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from typing import Dict, Any
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import copy
import json
import time
import os
import random
import pickle
import re
#baseline_bart = torch.hub.load('pytorch/fairseq', 'bart.large')
#baseline_bart.eval()
HOST_NAME = 'localhost'
PORT = 8080

SHARED: Dict[Any, Any] = {}
STYLE_SHEET = "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.1/css/bulma.css"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.3.1/js/all.js"
WEB_HTML = """
<html>
    <link rel="stylesheet" href={} />
    <script defer src={}></script>
    <head><title> Interactive Run </title></head>
    <body>
        <div class="columns" style="margin-top: 15px; margin-right:10px; margin-left:10px; height: 100%">
            <div class="column is-three-fifths ">
              <b>Describe the image below</b><br>
              <div class="column is-offset-one-quarter" style="height: 400px;">
                <img id="image-source" style="max-height: 100%; margin-left: auto; margin-right: auto;" src="not-pic.png" alt="Some image should be here" />
              </div>
              <b>Enter your text</b>
              <section class="hero is-info is-large has-background-light has-text-grey-dark" style="height: 150px">
                <div class="hero-foot column is-three-fifths" style="width:100%; height: 76px">
                  <form id = "interact">
                      <div class="field" style="width:100%; justify-content: space-between; text-align: left;">
                        <p class="control">
                          <textarea class="input" type="text" style="width:80%; height:100px;" id="userIn" placeholder="Type in a message"></textarea>
                          &nbsp; &nbsp; <button id="respond" type="submit" style="width=8%" class="button has-text-white-ter has-background-grey-dark">
                            Suggest
                          </button>
                          &nbsp; &nbsp; <button id="restart" type="reset" style="width=8%" class="button has-text-white-ter has-background-grey-dark">
                            Finish 
                          </button>
                          <div id="char-count" style="float:right">
                                Char Count: 0
                          </div>
                        </p>
                      </div>
                  </form>
                </div>
              </section>
              <br><br>
              <b> Select the suggestion that you like best: </b>
              1 <input type="radio" name="selection" id="selection_1" value="1" />
              2 <input type="radio" name="selection" id="selection_2" value="2" />
              3 <input type="radio" name="selection" id="selection_3" value="3" />
              Original Text <input type="radio" name="selection" id="selection_undo" value="last_iteration" />
              <br><br>
              <!-- div id="selected_candidate" style="height: 50px" -->
              <section class="hero is-info is-large has-background-light has-text-grey-dark" style="height: 300px">
                <div id="candidates" class="hero-body" style="overflow: auto; height: calc(300px); padding-top: 1em; padding-bottom: 0;">
                    <article class="media">
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <strong>Candidates</strong>
                          </p>
                        </div>
                      </div>
                    </article>
                </div>
              </section>
            </div>
            <div class="column">
              <div id="instructions" class="hero-body" style="float: left; overflow: auto; height: 500px; padding-top: 0; padding-bottom: 0;">
                  <p style="font-size:1.5em"><b>Instructions</b></em>
                  <ol style="padding-bottom: 10px;">
                      <li> <b>Describe the image (min 100 characters)</b> to the left in a descriptive/vivid manner. You're free to interpet it as you choose - try to be as imaginative as possible making use of metaphors, imagery and any other literary techniques you feel appropriate </li> <br>
                      <li> To assist you we have a model that can rewrite parts of your description. The model DOES NOT write for you, what you can do is write a sentence/paragraph in the inputbox and <b>mark out the area where you would like suggestions</b> bounded by a <b>start tag, '['; and an end tag, ']' and then hit SUGGEST</b>. You can also use an <b>_ to indicate a blank</b> the model should fill in. For example:
                          <ul>
                              <li> &emsp; <b> Input: </b> The New York skyline lies in the background surrounded by the ominous dark sky. But the Empire State Building shines bright, like a <b>[ purple torchlight ]</b>
                              <li> &emsp; <b> Output: </b> The New York skyline lies in the background surrounded by the ominous dark sky. But the Empire State Building shines bright, like a <b> beacon of hope in the darkness </b>
                          </ul></li><br>
                      <li> Once you hit SUGGEST you will receive 3 suggestions on the <b> bottom right pane </b>. Select one which you feel is appropriate you can continue <b>edit the chosen suggestion again</b> and use the model repeatedly. </li><br>
                      <li> Continue until you are happy with the description with a <b>minimum of 2 interactions with the system</b> (even if you reject the suggestions). Then <b>hit FINISH</b>.  
                  </ol>
              </div>
              <div id="parent" class="hero-body" style="float: left; overflow: auto; height: 600px; padding-top: 0; padding-bottom: 0;">
                <article class="media">
                  <div class="media-content">
                    <div class="content">
                      <p>
                        <strong>History</strong>
                        <br>
                        All interactions are logged here
                      </p>
                    </div>
                  </div>
                </article>
              </div>
            </div>
        </div>

        <script> 
            // const start;
            function load_image() {{
            document.getElementById("image-source").src = "pic.png";
            document.getElementById("image-source").height = "auto";
            // start = Date.now();
            }}
            load_image();
            var last_iteration = "";
            var count = 0;
            var uuid = create_UUID();
            var log_data = {{}};
            log_data["UUID"] = uuid;
            log_data["start"] = Date.now().toString();
            var log_count = 0;
            function createChatRow(agent, text) {{
                var article = document.createElement("article");
                article.className = "media"

                var figure = document.createElement("figure");
                figure.className = "media-left";

                var span = document.createElement("span");
                span.className = "icon is-large";

                var icon = document.createElement("i");
                // icon.className = "fas fas fa-2x" + (agent === "Older Iteration" ? " fa-user " : (agent === "Model" || agent.startsWith("Suggestion")) ? " fa-robot" : "");
                icon.className = "fas fas fa-2x" + (agent.startsWith("Older Iteration") ? " fa-user " : (agent === "Model" || agent.startsWith("Suggestion")) ? " fa-robot" : "");

                var media = document.createElement("div");
                media.className = "media-content";

                var content = document.createElement("div");
                content.className = "content";

                var para = document.createElement("p");
                if(agent.startsWith("Suggestion"))
                {{
                    var temp = "suggestion_";
                    var tid = temp.concat(agent.charAt(agent.length-1));
                    para.setAttribute("id", tid);
                }}

                var paraText = document.createElement("p")// document.createTextNode(text);
                paraText.innerHTML = text;

                var strong = document.createElement("strong");
                strong.innerHTML = agent;
                var br = document.createElement("br");

                para.appendChild(strong);
                para.appendChild(br);
                para.appendChild(paraText);
                content.appendChild(para);
                media.appendChild(content);

                span.appendChild(icon);
                figure.appendChild(span);

                if (agent !== "Instructions") {{
                    article.appendChild(figure);
                }};

                article.appendChild(media);

                return article;
            }}
            function LCS(a, b) {{
                var m = a.length,
                    n = b.length,
                    C = [],
                    i,
                    j;

                for (i = 0; i <= m; i++) C.push([0]);
                for (j = 0; j < n; j++) C[0].push(0);
                for (i = 0; i < m; i++) {{
                    for (j = 0; j < n; j++) {{
                        C[i+1][j+1] = a[i] === b[j]
                        ? C[i][j]+1
                        : Math.max(C[i+1][j], C[i][j+1]);
                    }}
                }}

                return (function bt(i, j) {{
	    if (i * j === 0) {{
		return ' ';
                    }}
                    if (a[i-1] === b[j-1]) {{
                        return bt(i-1, j-1) + " " + a[i-1];
                    }}
                    return (C[i][j-1] > C[i-1][j])
                            ? bt(i, j-1)
                            : bt(i-1, j);
                }}(m, n));
            }}
            document.getElementById('userIn').onkeyup = function () {{
                document.getElementById('char-count').innerHTML = "Char count: " + this.value.length;
            }};
            function create_UUID(){{
                    var dt = new Date().getTime();
                    var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {{
                    var r = (dt + Math.random()*16)%16 | 0;
                    dt = Math.floor(dt/16);
                    return (c=='x' ? r :(r&0x3|0x8)).toString(16);
                }});
                return uuid;
            }}
            function diff(newtext, oldtext){{ 
                text = '';
                oldwords = oldtext.split(' ');
                newwords = newtext.split(' ');
                lcs = LCS(oldwords, newwords);
                lcswords = lcs.split(' ');
                newwords.forEach(function(val, i){{
                if (! lcswords.includes(val))
                    text += "<b>"+val+"</b> ";  
                else
                    {{
                    text += val+" ";   
                    lcswords.splice(lcswords.indexOf(val), 1);
                    }}  
                }});
                return text 
            }}
            document.querySelectorAll('input[name="selection"]').forEach((elem) => {{
                elem.addEventListener("change", function(event) {{
                    var item = event.target.value;
                    log_data[log_count] = {{}};
                    log_data[log_count]['selection'] = item;
                    // log_data[log_count]['timestamp'] = Date.now().toString();
                    log_count+=1;
                    // alert(item);
                    if(item == "last_iteration")
                    {{
                        var tname = "suggestion_".concat(item)
                        // var temp = document.getElementById(tname);
                        // alert(temp.innerHTML)
                        // var output = document.getElementById("selected_candidate");
                        // output.innerHTML = last_iteration; // temp.innerHTML.substring(33);
                        document.getElementById("userIn").value = last_iteration ; // temp.innerHTML.substring(33).replace(/(<([^>]+)>)/ig, '');
                    }}
                    else 
                    {{
                        var tname = "suggestion_".concat(item)
                        var temp = document.getElementById(tname);
                        // alert(temp.innerHTML)
                        // var output = document.getElementById("selected_candidate");
                        // output.innerHTML = temp.innerHTML.substring(33);
                        document.getElementById("userIn").value = temp.innerHTML.substring(33).replace(/(<([^>]+)>)/ig, '');
                    }}
                }});
            }});
            document.getElementById("interact").addEventListener("submit", function(event){{
                event.preventDefault()
                var text = document.getElementById("userIn").value;
                if(!text.includes("[") && !text.includes("]") && !text.includes("_"))
                {{
                     alert("You haven't indicated where you want model suggestions with [, ] or _. Please read the instructions panel to clarify.");
                     return;    
                }}
                text = text.split("[").join("<replace>")
                text = text.split("]").join("</replace>")
                text = text.split("_").join("<mask>")
                text = text.charAt(0).toUpperCase() + text.slice(1);
                document.getElementById('userIn').value = "";
                var selection = window.getSelection();
                console.log(selection.toString())
                fetch('/interact', {{
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    method: 'POST',
                    body: text+"####"+selection.toString()
                }}).then(response=>response.json()).then(data=>{{
                    var parDiv = document.getElementById("parent");

                    count+=1
                    parDiv.append(createChatRow("Older Iteration ".concat(count.toString()), text));
                    var log_current = {{}}
                    log_current["input"] = text;
                    log_current["timestamp"] = Date.now().toString();
                    text = text.split("<replace>").join("");
                    text = text.split("</replace>").join("");
                    text = text.split("<mask>").join("");
                    document.getElementById("userIn").value = text;
                    last_iteration = text;
                    console.log(data)
                    parDiv.scrollTo(0, parDiv.scrollHeight);
                    
                    var suggestDiv = document.getElementById("candidates");
                    suggestDiv.innerHTML = '';
                    var temp = diff(data.beam_texts[0][0], text);
                    log_current["suggestion 1"] = temp;
                    suggestDiv.append(createChatRow("Suggestion 1", temp)); // data.beam_texts[0][0]));
                    temp = diff(data.beam_texts[1][0], text);
                    log_current["suggestion 2"] = temp;
                    suggestDiv.append(createChatRow("Suggestion 2", temp)); // data.beam_texts[1][0]));
                    temp = diff(data.beam_texts[2][0], text);
                    log_current["suggestion 3"] = temp;
                    suggestDiv.append(createChatRow("Suggestion 3", temp)); // data.beam_texts[2][0]));
                    var ele = document.getElementsByName("selection");
                    for(var i=0;i<ele.length;i++)
                        ele[i].checked = false;
                    log_data[log_count] = log_current;
                    log_count+=1

                }})
            }});
            document.getElementById("interact").addEventListener("reset", function(event){{
                event.preventDefault()
                if(count<2)
                {{
                    alert("Minimum number of interactions to complete the task is 2. You have ".concat(count.toString()));
                    return;
                }}
                var text = document.getElementById("userIn").value;
                if(text.length<100)
                {{
                    alert("The descriptions should be at least 100 characters long");
                    return;
                }}
                log_data["Final"] = document.getElementById('userIn').value;
                log_data["end"] = Date.now().toString();
                document.getElementById('userIn').value = "";
                load_image();
                console.log(JSON.stringify(log_data));
                var parDiv = document.getElementById("parent");

                parDiv.innerHTML = '';
                parDiv.append(createChatRow("Instructions", "All interactions are logged here"));
                
                fetch('/reset', {{
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    method: 'POST',
                    body: JSON.stringify(log_data)
                }}).then(response=>response.json()).then(data=>{{
                    log_data = {{}};
                    log_count = 0;
                }})
                var return_code = "TASK COMPLETE. The code for you to enter back is (make sure you copy this exactly): ".concat(uuid)
                alert(return_code);
            }});
        </script>

    </body>
</html>
"""  # noqa: E501


class MyHandler(BaseHTTPRequestHandler):
    """
    Handle HTTP requests.
    """

    def _interactive_running(self, opt, reply_text, model_idx=0):
        reply = {'episode_done': False, 'text': reply_text}
        print("reply: ", reply)
        if model_idx == 1:
            reply['text'] = re.sub("<replace>.*?</replace>", "<mask>", reply['text'], flags = re.DOTALL)
            beams = bart.fill_mask([reply['text']], topk=3, beam=5)
            model_res = {}
            model_res['beam_texts'] = []
            for i in range(len(beams[0])):
                model_res['beam_texts'].append((beams[0][i][0], beams[0][i][1].numpy().item()))
            print("Using baseline")
        else:
            SHARED['agent'].observe(reply)
            model_res = SHARED['agent'].act()
            print("Using finetuned")
        return model_res #, baseline_res

    def do_HEAD(self):
        """
        Handle HEAD requests.
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        """
        Handle POST request, especially replying to a chat message.
        """
        if self.path == '/interact':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            logging.info("POST request,\nPath: "+str(self.path)+"\nHeaders:\n"+str(self.headers)+"\n\nBody:\n"+body.decode('utf-8')+"\n")
            #print("Received: ", body.decode('utf-8'))
            temp = body.decode('utf-8')
            temp = temp.split("####")
            assert len(temp) == 2
            index = temp[0].find(temp[1])
            if index == -1 or len(temp[1]) == 0:
                ip = temp[0]
            else:
                ip = temp[0][:index] + "<mask>" + temp[0][index+len(temp[1]):]
            #print("Input: ", ip)
            try:
                params = self.headers['Referer'].split("?")[-1]
                if "&" in params:
                    params = params.split("&")[0]
                assert "=" in params, print("Params Error ", params)
                temp = params.split("=")
                assert len(temp) == 2, print("Params Length Error ", temp)
                model_idx = int(temp[-1])
            except:
                model_idx=0
            model_response = self._interactive_running(
                SHARED.get('opt'), ip, #body.decode('utf-8')
                model_idx
            )
            print("Model response: ", model_response)
            #print("Baseline response: ", baseline_response[0][0])
            #print("Model response: ", model_response)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            json_str = json.dumps(model_response)
            self.wfile.write(bytes(json_str, 'utf-8'))
            SHARED['agent'].reset()
        elif self.path == '/reset':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            logging.info("POST request,\nPath: "+str(self.path)+"\nHeaders:\n"+str(self.headers)+"\n\nBody:\n"+body.decode('utf-8')+"\n")
            #print("Received: ", body.decode('utf-8'))
            temp = body.decode('utf-8')
            temp = json.loads(temp)
            print("Received complete logs ", temp)
            pickle.dump(temp, open("./Process-Results/"+temp['UUID']+".pkl", "wb"))
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            SHARED['agent'].reset()
            self.wfile.write(bytes("{}", 'utf-8'))
        else:
            return self._respond({'status': 500})

    def do_GET(self):
        """
        Respond to GET request, especially the initial load.
        """
        logging.info("GET request,\nPath: "+str(self.path)+ "\nHeaders:\n"+str(self.headers)+"\n")
        
        paths = {
            '/': {'status': 200},
            '/favicon.ico': {'status': 202},  # Need for chrome
            '/pic.png': {'status':200},
        }
        if self.path in paths:
            self._respond(paths[self.path])
        else:
            self._respond({'status': 500})

    def _handle_http(self, status_code, path, text=None):
        self.send_response(status_code)
        dir_name = "./Image-Source/"
        if path[-3:] == "png":
            try:
                params = self.headers['Referer'].split("?")[-1]
                if "&" in params:
                    params = params.split("&")[-1]
                assert "=" in params, print("Params Error ", params)
                temp = params.split("=")
                assert len(temp) == 2, print("Params Length Error ", temp)
                idx = int(temp[-1])
                filename = os.listdir(dir_name)[idx%(len(os.listdir(dir_name)))]
            except:
                filename = random.choice(os.listdir(dir_name))
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            image_file = open(dir_name+"/"+filename, "rb")
            file_data = image_file.read()
            image_file.close()
            return file_data
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = WEB_HTML.format(STYLE_SHEET, FONT_AWESOME)
        return bytes(content, 'UTF-8')

    def _respond(self, opts):
        wait()
        response = self._handle_http(opts['status'], self.path)
        self.wfile.write(response)


def setup_interweb_args(shared):
    """
    Build and parse CLI opts.
    """
    parser = setup_args()
    parser.description = 'Interactive chat with a model in a web browser'
    parser.add_argument('--port', type=int, default=PORT, help='Port to listen on.')
    parser.add_argument(
        '--host',
        default=HOST_NAME,
        type=str,
        help='Host from which allow requests, use 0.0.0.0 to allow all IPs',
    )
    return parser


def shutdown():
    global SHARED
    if 'server' in SHARED:
        SHARED['server'].shutdown()
    SHARED.clear()


def wait():
    global SHARED
    while not SHARED.get('ready'):
        time.sleep(0.01)


def interactive_web(opt):
    global SHARED

    opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()
    SHARED['opt'] = agent.opt
    SHARED['agent'] = agent

    SHARED['world'] = create_task(SHARED.get('opt'), SHARED['agent'])

    MyHandler.protocol_version = 'HTTP/1.0'
    httpd = ThreadingHTTPServer((opt['host'], opt['port']), MyHandler)
    SHARED['server'] = httpd
    logging.info('http://{}:{}/'.format(opt['host'], opt['port']))

    try:
        SHARED['ready'] = True
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()


@register_script('interactive_web', aliases=['iweb'], hidden=True)
class InteractiveWeb(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_interweb_args(SHARED)

    def run(self):
        return interactive_web(self.opt)


if __name__ == '__main__':
    InteractiveWeb.main()
