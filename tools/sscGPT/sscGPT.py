if search_type == "ASI Query from URL":
    url = st.sidebar.text_input("", placeholder="Enter URL and press enter")
    #send_to_asi = st.sidebar.checkbox("Check this box to send search to ASI")
    generated_text_chunks = []
    if url:

        try:
            chunk_size = 2800
            response = requests.get(url)
            response.raise_for_status()
            parsed_text = parse_html_to_text(response.text)

            num_chunks = len(parsed_text) // chunk_size + (len(parsed_text) % chunk_size > 0)
            st.warning(f"`{num_chunks}` x `{chunk_size}` token (word) packages will be submitted to OpenAI model: `text-davinci-003`")
            completions = None # Initialize completions
            for i in range(0, len(parsed_text), chunk_size):
                chunk = parsed_text[i:i+chunk_size]
                chunk_length = len(chunk)
                prompt_template = "Read contents of {}, parse for indicators and use as data {}. Do not print search results."
                prompt = prompt_template.format(chunk, persona_asi)
                completions = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=2024,
                    n=1,
                    stop=None,
                    temperature=1.0,
                )
                generated_text_chunks.append(completions.choices[0].text.strip())
            total_size =  num_chunks * chunk_size
            col1, col2, col3 = st.columns(3)
            col1.metric("HTML Word Count", total_size,total_size )
            col2.metric("Token Packages", num_chunks,num_chunks )
            col3.metric("Total Prompt Token Count", len(prompt), len(prompt))
            st.markdown("----")
            st.info("Generated Attack Surface Intelligence Query from URL")


        except requests.exceptions.RequestException as e:
            st.error(f"Error occurred while fetching the URL: {e}")

        generated_text = '\n'.join(generated_text_chunks)
        query = completions.choices[0].text.strip() if completions is not None else ""
        if st.checkbox("Check this box to send search to ASI"):
            query_str = generated_text_chunks[0].split(":")[1].strip() # Extract the query string
            #st.write(f"Query: {query_str}")
            assets = search_assets(query_str)
            st.write(assets)
        st.write(f"{generated_text_chunks[0]}")
