import os
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES


# =============================================================================
# 1. FUNÇÕES DO MODELO
# =============================================================================
def preparar_entrada_imagem(objeto_imagem):
    img = objeto_imagem.convert('RGB')
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    vetor_imagem = img_array.reshape(1, 12288)
    return vetor_imagem


def executar_diagnostico(modelo_carregado, objeto_imagem):
    dados_formatados = preparar_entrada_imagem(objeto_imagem)
    probabilidade = modelo_carregado.predict(dados_formatados, verbose=0)[0][0]

    limiar = 0.5
    if probabilidade >= limiar:
        resultado = "ALERTA: Lesão com características de Melanoma/BCC"
        cor = DANGER_COLOR
    else:
        resultado = "BENIGNO: Baixo risco identificado"
        cor = SUCCESS_COLOR

    return resultado, probabilidade, cor


# =============================================================================
# 2. CONFIGURAÇÕES DE ESTILO (PALETA DE CORES MODERNA)
# =============================================================================
BG_APP = "#F0F2F5"  # Cinza muito claro (fundo da janela)
BG_CARD = "#FFFFFF"  # Branco (fundo dos painéis)
BG_DROP = "#E4E6EB"  # Cinza para a área de soltar a imagem
TEXT_DARK = "#1C1E21"  # Preto suave para textos principais
TEXT_MUTED = "#65676B"  # Cinza para subtítulos
PRIMARY = "#0064E0"  # Azul moderno para botões normais
PRIMARY_HOVER = "#0053BA"  # Azul escuro para hover
ANALYZE_BTN = "#00A400"  # Verde para o botão principal
ANALYZE_HOVER = "#008C00"  # Verde escuro para hover
SUCCESS_COLOR = "#00A400"
DANGER_COLOR = "#E41E3F"
FONT_MAIN = ("Segoe UI", 10)
FONT_TITLE = ("Segoe UI", 16, "bold")


# =============================================================================
# 3. INTERFACE GRÁFICA (GUI) COM DRAG & DROP E ESTILO MODERNO
# =============================================================================
class AppDiagnostico:
    def __init__(self, root):
        self.root = root
        self.root.title("PROJETO DE AI - Diagnóstico de Pele")
        self.root.geometry("550x700")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_APP)

        # Configura a janela para aceitar arquivos arrastados
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.soltar_imagem)

        self.modelo = None
        self.imagem_pil = None
        self.imagem_tk = None

        self.construir_interface()

        # Inicia o carregamento do modelo após a interface abrir
        self.root.after(100, self.carregar_modelo)

    def construir_interface(self):
        # --- HEADER ---
        frame_header = tk.Frame(self.root, bg=BG_APP)
        frame_header.pack(fill=tk.X, pady=(20, 10))

        lbl_titulo = tk.Label(frame_header, text="PROJETO DE IA", font=FONT_TITLE, bg=BG_APP, fg=TEXT_DARK)
        lbl_titulo.pack()

        lbl_sub = tk.Label(frame_header, text="Análise inteligente de lesões na pele", font=("Segoe UI", 11),
                           bg=BG_APP, fg=TEXT_MUTED)
        lbl_sub.pack()

        # --- CARD PRINCIPAL ---
        self.card = tk.Frame(self.root, bg=BG_CARD, padx=20, pady=20)
        self.card.pack(expand=True, fill=tk.BOTH, padx=30, pady=(0, 20))

        # Status do Modelo
        self.lbl_status = tk.Label(self.card, text="Carregando Inteligência Artificial...",
                                   font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=PRIMARY)
        self.lbl_status.pack(pady=(0, 15))

        # Área da Imagem (Dropzone)
        self.canvas = tk.Canvas(self.card, width=320, height=320, bg=BG_DROP, highlightthickness=0, cursor="hand2")
        self.canvas.pack(pady=10)

        # Textos do Canvas (Instruções)
        self.txt_canvas1 = self.canvas.create_text(160, 145, text="Arraste uma imagem para cá", fill=TEXT_DARK,
                                                   font=("Segoe UI", 12, "bold"), justify="center")
        self.txt_canvas2 = self.canvas.create_text(160, 175, text="Ou clique para selecionar um arquivo",
                                                   fill=TEXT_MUTED, font=("Segoe UI", 10), justify="center")

        # Vincula o clique no canvas para abrir arquivo
        self.canvas.bind("<Button-1>", lambda e: self.carregar_imagem())

        # Botão de Análise (Estilo Moderno)
        self.btn_analisar = tk.Button(
            self.card, text="ANALISAR IMAGEM", font=("Segoe UI", 11, "bold"),
            bg=ANALYZE_BTN, fg="white", activebackground=ANALYZE_HOVER, activeforeground="white",
            relief="flat", cursor="hand2", pady=10, state=tk.DISABLED, command=self.analisar
        )
        self.btn_analisar.pack(fill=tk.X, pady=(20, 5), padx=40)

        # Efeitos de Hover para o botão
        self.btn_analisar.bind("<Enter>",
                               lambda e: self.hover_in(self.btn_analisar, ANALYZE_HOVER) if self.btn_analisar[
                                                                                                'state'] == tk.NORMAL else None)
        self.btn_analisar.bind("<Leave>", lambda e: self.hover_out(self.btn_analisar, ANALYZE_BTN) if self.btn_analisar[
                                                                                                          'state'] == tk.NORMAL else None)

        # --- RESULTADOS ---
        self.lbl_resultado = tk.Label(self.card, text="", font=("Segoe UI", 12, "bold"), bg=BG_CARD, wraplength=400)
        self.lbl_resultado.pack(pady=(15, 0))

        self.lbl_certeza = tk.Label(self.card, text="", font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT_MUTED)
        self.lbl_certeza.pack(pady=5)

    def hover_in(self, widget, color):
        widget['bg'] = color

    def hover_out(self, widget, color):
        widget['bg'] = color

    def carregar_modelo(self):
        nome_modelo = "modelo_mlp_evoluido.keras"
        if not os.path.exists(nome_modelo):
            self.lbl_status.config(text="Erro: Modelo não encontrado!", fg=DANGER_COLOR)
            messagebox.showerror("Erro de Arquivo",
                                 f"O arquivo '{nome_modelo}' precisa estar na mesma pasta deste script.")
            return

        try:
            self.modelo = tf.keras.models.load_model(nome_modelo)
            self.lbl_status.config(text="IA Pronta para uso", fg=SUCCESS_COLOR)
        except Exception as e:
            self.lbl_status.config(text="Erro ao carregar IA", fg=DANGER_COLOR)
            messagebox.showerror("Erro Fatal", f"Falha ao carregar o modelo Keras:\n{e}")

    def atualizar_imagem_tela(self, img):
        self.imagem_pil = img

        # Recorta e redimensiona mantendo um visual quadrado preenchido (CROP)
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        img_cropped = img.crop((left, top, left + min_dim, top + min_dim))
        img_preview = img_cropped.resize((320, 320), Image.Resampling.LANCZOS)

        self.imagem_tk = ImageTk.PhotoImage(img_preview)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imagem_tk)

        # Limpa resultados anteriores e habilita o botão de análise
        self.lbl_resultado.config(text="")
        self.lbl_certeza.config(text="")

        if self.modelo:
            self.btn_analisar.config(state=tk.NORMAL, bg=ANALYZE_BTN)

    def carregar_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecione uma imagem da lesão",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.webp"), ("Todos os arquivos", "*.*")]
        )
        if caminho:
            self._abrir_e_exibir(caminho)

    def soltar_imagem(self, event):
        caminhos = self.root.tk.splitlist(event.data)
        if caminhos:
            self._abrir_e_exibir(caminhos[0])

    def _abrir_e_exibir(self, caminho):
        try:
            img = Image.open(caminho)
            self.atualizar_imagem_tela(img)
        except Exception as e:
            messagebox.showerror("Erro de Leitura", f"Não foi possível abrir esta imagem:\n{e}")

    def analisar(self):
        if self.imagem_pil and self.modelo:
            # Feedback visual de carregamento
            self.lbl_resultado.config(text="Processando imagem...", fg=PRIMARY)
            self.lbl_certeza.config(text="")
            self.root.update()

            # Executa a predição
            resultado, certeza, cor = executar_diagnostico(self.modelo, self.imagem_pil)

            # Atualiza a interface
            self.lbl_resultado.config(text=resultado, fg=cor)
            self.lbl_certeza.config(text=f"Probabilidade calculada: {certeza * 100:.2f}%")


# =============================================================================
# 4. INICIALIZAÇÃO
# =============================================================================
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    janela_principal = TkinterDnD.Tk()
    app = AppDiagnostico(janela_principal)
    janela_principal.mainloop()