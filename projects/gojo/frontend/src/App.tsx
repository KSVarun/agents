import { useRef, useState } from "react";
import "./App.css";
import clsx from "clsx";
import axios from "axios";

function App() {
  const [data, setData] = useState<{
    currentMessage: string | null;
    messages: string[];
    loading: boolean;
  }>({
    currentMessage: null,
    messages: JSON.parse(localStorage.getItem("chatMessages") || "[]"),
    loading: false,
  });
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  function sendMessage(message: string) {
    setData((data) => ({ ...data, loading: true }));
    axios
      .post("http://localhost:8090/echo", { query: message })
      .then((res) => {
        setData((data) => {
          const updatedMessages = [...data.messages, res.data.message];
          localStorage.setItem("chatMessages", JSON.stringify(updatedMessages));
          return {
            ...data,
            loading: false,
            messages: updatedMessages,
          };
        });
      })
      .catch((err) => {
        console.log(err);
        setData((data) => ({ ...data, loading: false }));
      });
  }

  function textAreaIncreaseOnEnterPress() {
    if (!textareaRef.current) {
      return;
    }
    textareaRef.current.style.height =
      5 + textareaRef.current.scrollHeight + "px";
  }

  function textAreaIncreaseOnLargeContentType() {
    if (!textareaRef.current) {
      return;
    }
    // const loop = Math.ceil(textareaRef.current.scrollHeight / 38);
    // if (loop > 1) {
    //   for (let i = 0; i < loop; i++) {
    //     textAreaIncreaseOnEnterPress();
    //   }
    // }
  }

  function textAreaReset() {
    if (!textareaRef.current) {
      return;
    }
    textareaRef.current.style.height = "21px";
  }

  return (
    <>
      <div className="container">
        <section className="messagesSection">
          {data.messages.map((message, idx) => {
            return (
              <div
                key={`${idx}+${message}`}
                className={clsx(
                  "message",
                  idx === 0 || idx % 2 === 0 ? "userMessage" : "backendMessage"
                )}
              >
                {message}
              </div>
            );
          })}
        </section>
        <footer>
          <form
            onSubmit={(e) => {
              if (data.loading) {
                return;
              }
              e.preventDefault();
              if (
                data.currentMessage &&
                data.currentMessage.trim().length > 0
              ) {
                const updatedMessages = [...data.messages, data.currentMessage];
                localStorage.setItem(
                  "chatMessages",
                  JSON.stringify(updatedMessages)
                );
                sendMessage(data.currentMessage);
                setData((data) => ({
                  ...data,
                  messages: updatedMessages,
                  currentMessage: null,
                }));
              }
            }}
          >
            <textarea
              ref={textareaRef}
              disabled={data.loading}
              rows={1}
              value={data.currentMessage ? data.currentMessage : ""}
              onKeyDown={(e) => {
                if (data.loading) {
                  return;
                }
                if (e.key === "Enter" && e.shiftKey) {
                  textAreaIncreaseOnEnterPress();
                  return;
                }
                if (
                  e.key === "Enter" &&
                  data.currentMessage &&
                  data.currentMessage.trim().length > 0
                ) {
                  sendMessage(data.currentMessage);
                  setData((data) => ({
                    ...data,
                    messages: [...data.messages, data.currentMessage!],
                    currentMessage: null,
                  }));
                  textAreaReset();
                  return;
                }
                textAreaIncreaseOnLargeContentType();
              }}
              onChange={(e) => {
                if (e.target.value.trim() === "") {
                  textAreaReset();
                  setData((data) => ({
                    ...data,
                    currentMessage: e.target.value.trim(),
                  }));
                  return;
                }
                setData((data) => ({
                  ...data,
                  currentMessage: e.target.value,
                }));
              }}
            />
            <button type="submit">Send</button>
          </form>
        </footer>
      </div>
    </>
  );
}

export default App;
