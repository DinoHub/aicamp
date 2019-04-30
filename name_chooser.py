import tkinter as tk

class Name_chooser(tk.Frame):
    def __init__(self, parent, names):
        self.parent = parent
        tk.Frame.__init__(self, parent, bg='#81ecec')
        self.chosen = None
        # create a prompt, an input box, an output label,
        # and a button to do the computation
        self.prompt = tk.Label(self, text="Choose the name:",
                               anchor="w",
                               font=('Ubuntu Mono',15), 
                               fg='#000000',
                               bg='#81ecec',)
        self.names = names
        self.name_buttons = []
        for i, name in enumerate(names):
            self.name_buttons.append(
                tk.Button(self, text=name, 
                          font=('Ubuntu Mono',20), 
                          bg='#00b894',
                          activebackground='#0dd8b0',
                          width=20,
                          command = lambda i=i: self.choose(i)))
        self.custom = tk.Entry(self, font=('Ubuntu Mono',20), 
                               justify='center')
        self.submit_custom = tk.Button(self, 
                    text='Submit custom name', 
                    font=('Ubuntu Mono',20), 
                    bg='#a29bfe',
                    activebackground='#0dd8b0',
                    width=20,
                    command = self.custom_choose)
        # self.output = tk.Label(self, text="")

        global_pady = 4
        global_padx = 20
        # lay the widgets out on the screen. 
        self.prompt.pack(side="top", fill="x", pady=(5,global_pady), padx=global_padx)
        # self.output.pack(side="top", fill="x", expand=True)
        for button in self.name_buttons:
            button.pack(side="top", fill="x", pady=global_pady, padx=global_padx)
        self.custom.pack(side="top", fill="x", pady=(global_pady,0), padx=global_padx)
        self.submit_custom.pack(side="top", fill="x", pady=(0,15), padx=global_padx)
        self.chosen_name = None

    def choose(self, idx):
        self.chosen_name = self.names[idx]
        print('Chosen name:{}'.format(self.chosen_name))
        self.parent.destroy()

    def custom_choose(self):
        self.chosen_name = self.custom.get()
        print('Chosen custom name:{}'.format(self.chosen_name))
        self.parent.destroy()


# if this is run as a program (versus being imported),
# create a root window and an instance of our example,
# then start the event loop

def choose_name(classes=['Tom', 'Dick', 'Harry'], frame_size=None,screen_loc=[0,0],  screen_size=None, bb_loc=None):
    root = tk.Tk()
    root.configure(bg='#81ecec')
    ws = root.winfo_screenwidth()  # width of screen
    hs = root.winfo_screenheight() # height of screen
    est_width = 400
    print(ws, hs)
    # frame_h, frame_w = frame_size
    # screen_w, screen_h = screen_size
    if bb_loc:
        ## TODO: this is iffy, bb_loc does not take into account scaling done if max window is hit.
        x = screen_loc[0] + bb_loc['rect']['r']
        if (x+est_width) > ws:
            x = screen_loc[0] + bb_loc['rect']['l'] - est_width
            print('HERE')

        # y = screen_loc[1] + bb_loc['rect']['t']
        y = screen_loc[1] + bb_loc['rect']['t']
    else:
        x = screen_loc[0]
        y = screen_loc[1]

    x = screen_loc[0]
    y = screen_loc[1]

    root.geometry('+%d+%d'%(int(x),int(y)))
    choser = Name_chooser(root, classes)
    choser.pack(fill="both", expand=True)
    root.mainloop()
    # print('out of loop Chosen:{}'.format(choser.names[choser.chosen]))
    chosen_name = choser.chosen_name
    if chosen_name is not None and chosen_name not in classes:
            classes.append(chosen_name)
    return chosen_name, classes

if __name__ == '__main__':
    choose_name()