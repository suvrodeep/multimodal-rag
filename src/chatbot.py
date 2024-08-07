import streamlit as st
from answer_generator import Generation


# Get Multimodal RAG Chain
@st.cache_resource
def get_chain():
    generation = Generation()

    return generation.multi_modal_rag_chain()


# Response generator function
def generate_response(input_text: str, chain):
    if input_text is None:
        return "No query provided"
    else:
        return chain.invoke(input_text)


def main():
    rag_chain = get_chain()

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input("Please input query here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(input_text=prompt, chain=rag_chain)
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
