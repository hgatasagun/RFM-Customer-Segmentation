# FLO Customer Segmentation using the RFM Method

![Açıklama metni](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAPEA8PEBAQEBAQDQ8QDg8ODhAPDhAQFRUWFhYVFRUYHSggGBolGxYVITEhJSkrLi4uGB8/ODMsNygtLisBCgoKDg0OGxAQGzAlICUtLS0tLS0uLS0tLS0vLy0tLS0tLS0tLS0tLS0tLystLS0tLS0tKy0tLS0tKy0tLS0tK//AABEIAJEBWwMBIgACEQEDEQH/xAAcAAEAAQUBAQAAAAAAAAAAAAAAAQIDBAUGBwj/xABGEAABAwICBAoIAwYFBAMAAAABAAIDBBESIQUxQVEGExQiMmFxkaGxBxVSYnKBktEWI0IzorKzwfA1U4LC8WNzw+EIJEP/xAAaAQACAwEBAAAAAAAAAAAAAAAAAQIDBAUG/8QANBEAAgECAwINBAIDAQAAAAAAAAECAxEEEiExUQUTFEFSYXGBkaGx0fAiMjPBFeFCkvFi/9oADAMBAAIRAxEAPwD2FVKApQBClEQIKVClAEIiIEEUqEDClQiBEoilAEKURMAiIgQREQAUqECAJREQIIiIAKpUqpABaXhpRGo0dXQtF3Po5sA3vDS5o7wFukQI+PwbqVtuFuiTRV1XS2sI538X/wBp3Pj/AHHNWpWkotYIiqjCYiWNsvYP/j/X86vpidbYZ2DsxMef5a8hXa+h6u4jS8AOQnjmgJ2Ztxt/ejaPmozX0sdN/Wj3nSWixIS9lg/aNjvsVopYnMOFzSDuIXZLw30m8MeWTCnp3nk8Djd7DYTSai641sGYG/M7lgqRS1OzhITqyyrZv3HoVPTPkNmNJ3nYO0rodG6PbCLnnPIzdsHUF5x6JeF97aOqHEu5xpZHG5cM3OiJ3jMjquNgv6snTitpDFKdObg/+o8P9P0l6uiZtbSyO+p9v9q8qe3avQ/ThUCTSjQDcR0ULfmXyO/3BefLdD7Ucmo/rZaRS8KFIR9dIiLIbAoKlU4BcutmQATvAvbzKAew5aXTs4JHNycQOYN6j19Pvb9AWun6Tvjd5qhY88t56pYSg/8ABeCNp6+n3t+gJ6+n3t+gLVojPLePkdDoLwRtDp+fez6ArFJwplmbiYW2vbOPCcwCDY7C0gg7QQtXM+IyRQyv4uJ5JmeQ63EttiYCP1OuG9hcdiv6Wq6bFHLDK1z3yywyRtBN4jI8wPGzmNIbb2XDVhslnltuQeGoKSjxa158unVzdXpvNp6+qN7foCevp97foC1ihPPLeT5JQ6C8EbX1/Ub2/QE9f1G9v0BapEZ5bw5HQ6C8EbObhBUA2BZ0GnoDaAVR+Iqjez6AtfUax8DP4QrK6MEsqPNVYpTkkud+rNt+Iqjez6An4iqN7PoC1KKWVFdkbb8RVG9n0BPxFUb2fQFqURlW4LI234iqN7PoCfiKp3s+gLUojKgsjbfiOp3s+gJ+I6jez6AtSiMqCyN4zT9QWg3bfE4dAbA37qr19Pvb9AWpi6A+N/k1Sunh6FOVNNxXxs4+JqzjVai2l/SNp6+n3t+gJ6/n3s+gLCpaOSW+BuLDa/Oa3X2lXpdFzMaXOYA1ouTjjOXyKm6eHTtZX7ipTrNXTdi/6/n3t+gLc6BrnzNkL7c1zQLC2sFckul4KdGX4m+RVOLo040m4pJ6epZhqs5VEmzzn086CsafSLBkRyaoI2EXdE4/vtv8K8jX1bwj0OyupZ6STozRloda+B4zY8dbXAH5L5YraSSCWSCVuGSKR0cjdzmmxtvHWufTeljfNc5ZVTFSFdCsRUws3Q9byeop6jVxNRFIbey1wJ8LrCQhMR9Yce61r5f07V4z6TuDzaWdtRFYR1TnkxjLBILF1h7JvfqN+pel8EK7lFBRzE3L6aPGffaMLv3mlef+mGpxVFND7EBee17reTFz6i01O9wdJ8erPRp37LX9bF/0RaIa501a6xLDxMQ2teWhz3dWRAHa5eqPmc4WJy8+1eXehyrzq4D/ANKZo6+c13+xemBFP7SOPu68r9Vuyx89+kuo4zStYb3DXxxjqwRsBH1YlzCzdNVfH1NTNe4lqZpAfdc8keFlhLelZWOJJ3bZbcoVxxVtA7n10iIshsCK1HO1znNBu5lsQscrq4Hi5btABI6je3kU7Cb0OLlpec7nDpHYd6p5L7w7isqXpO+I+aoXH46R6jjZ7yxyX3h3FOS+8O4q+iOOkHGz3ljkvvDuKnkvvjuKvIjjpBxs95Y5L7w7inJfeHcVfRHHSDjZ7yxyX3h3FTyX3h3FXkRx0g42e819W2z7a7NZ/CFZWTX9M/Az+ELGXcpu8F2L0PPVPvl2v1CIimQM7RlBx5IAxnFEHtLnWMRLg7K9hrFyM+5bSLgy+7cUcZtJM1xLWHFGB+XfeL4rDZiWjpa2SBwlisXC4wve5kbgRbn4QSQNdt4Cn1vW2GdPzWxN11jjhjOJl3GW5NzmdbtRuMlTK97JF0I3V89vHq9NGutIzqzQroow+SMANgDpCCWkymSzGlzTckAtO6wIWqV+XSdTK0sm4u3GPk/JknIc55vZzJC6wGyzrDYFYU4NtXasQmsrte/y3okERFMgZEXQHxv8mqVEXQHxv8mqV18N+Jd/qzhYv80u70RqOEccpY0Ma6aNrmSGDE6wLQ8Hmg5XDhqF7t2rRcG45C9r4Y5IsJnZynjec3EMAs4Bt8LSchne+w5dzT8WHDjMRjzxBuPGcsrFgJGarkio2RltMyRhxl3ONQQb3vYvGEZ2OveuZiMLCWKi23rt0W3atbb738urv4LhetS4OnTilpond7LWd1mtsslor63122F0nBPoy/E3yK5pdLwT6MvxN8iuhjfwvu9Tz+E/KjfLxf05cGML2aTibzX4YasADJ+qOQ9o5h7Gb17QsPSlBHVQy08zcUU0bmSN22O47CNYOwgLixdnc6zVz5LVcZWw4S6El0fVTUkubo3cx9rCSM5seO0dxBGxa1pWhMoaLiIikRPcPQ1W8Zo4xbYKmVn+l9pB4vd3LiPSJVcbpKp3RlkbeoMY2/7xctn6Ea4NnrICcnwMmA2fluwuPdI3uXJaSqeOnmm/zJZH/W4u/qsGJ0Z6HgZXbluVvP2TOl9F1ZxekGNOqaGWP52xj+HxXrHCCs5PSVU/+VTTPHaGG3jZeE8HKviKull1BlRGXH3S8B37pK9T9LdbxWjJW3sZ5YoR187G4fSwqNDXTrDhWOWebevNfEeCtFgB1KUQldE88UyFUIiiSPrpERZTYYlKDxk172JBAJdlr2HV4/LUsxYNI1rZJA1hF9b8QIdYnIjWLG43ZLMUntI8xyUvSd8R81Sqpek74j5qlcE9KEREAVQQ8YHOdJFCyOUBzpZQxuQBt22KippnRRtlY+OeIzEOfDIHtaHl1rnZmWhc/WcEIaiZ0kks2KR7cWEssNlhduq1u5QzgXDTzXZLPiilOEkssSw5X5uq4Vn05dnmW5afT7svlfNfb3dR0CIirKgiIgZg1x55+Fn8IWPddfStGBmQ6I2BXcI3N+kLauFIQWVxemnMcedFuTd+dnF3S67KUtYx0hbzW2xkNFmgkC56he56gVQ2VheWWGIA/pFsrXHaMTTn7QUnwtBK+R+QuTyOQupuuzwjc36QtRpV5EpAyGFuVrBW0eEYVW0ovyMmMqcmpqctdbadjf6NFdFseNdv8AnGu3+AWjj1uOZ/Lw6L8TXItjxrt/gE492/wCOPW4P5eHRfiY8XQHxv8mosjlDt/gU45/tea20uEVCCjl8+vsMFbExqTcrPX2LFksr/ABz/AGvNOOf7XmrP5VdB+P8ARVxsfn/THXScE+jL8TfIrR8c/wBrzXQ8GHkskuSec3X2Kqrj1Vjky2v1/wBGrB1E6yS6/m03SIiyHYOC9LfBD1hS8fC29XStc5gGuaLW+PrP6m9YI/UvnwFfX68F9MHA7kc/LoG//WqXnjWtGUNQcz2NfmR13G0BWU5cxGavqeesKqVppV1XopZt+C+lHUlQZW63U1VD85I3Bv7+A/JRay1bHYSD2eK2h29qw4tapnoeA5LJOPPdP19mRZdp6WNMcdT6KjB/aU/KnjbdzWtb/wCRcWrGlax0row7/wDGFkDPgaXOA73lLC/eWcNfhi+u3lf9GCqJDsVatkrczzSIRESGfXSlQiymwwqT9rPcDWw3xXJGdri+XcFmAm5FsrAg31nO4t8h3rDpGfnTOz2C9iAde29j8tXaVmqctvgRWw5KXpO+I+aoWa+izPO2lU8i95cjktbond5XR6Xr7GIiy+Re8nIveRyWt0WHK6HS9fYxmusQdxBSV+JznHW5xcbark3WTyL3lPIveRyata2Vi5XQvfMvP2MNFmci97wUci97wRyWt0WPllDpLz9jERZnIve8P/aci95HJa3RYcrodL19jOpOgz4QrWlI5nQyNp3iOYt/Kc62HF1ggjPMatquRyNY1rTckNGoC3mp5S33u4fdZJYHEZm1B7er3MLxtBSvmW0tcGdJNqaGJswJc6J0NRibhJcLsfiGWdhmppYDaN8n7bi2iSx5pfhAcVd5S33u4fdRylvvdw+6tq0MXUSThs2bBPGYfNJxkkm72u9Pn6RfWj0z+1PwN/otrylvvdw+61OlXAyXGrANfYrMJhqtKTlONla3mjk8MV6dSglCV/qXozCRSi3HmiFYp6CaerDIXRtcKF1+PDi3AJm6sNs72WQrVDE19ZZ1IKwChcRGRCcJ40c/81wHVlnmoyV7dp1eBoRniss1o4yve1tj3tLzRGk9GzQT05ldC7GyfCIBILWDL3xnrCo5dHxnE42Y8GLDcb7Yd19ttdlf0zTNjnpsNCKO8dRcg0/5lgzL8tx1de9a40sfKrYWZ0zpiMI/a8c04u2+Sjqtm/8AS7DZwhQw/HKLukqTay5enUfSat/5T1v9ytrskRSrDzxC6Tgt0JPib5LnF0fBboSfE3yUofcbMB+ddj9DdqFKhXHeCxNK6PiqoZaeZuOKVhZI3VkdoOwjWDsICy1q9I6QfG/C0CwAOYve6Tko6k4QcnZHzdww4Ny6Mqn00l3N6cEtrCaInJ3bsI2EbrLUMK+hOGujItK0phkbxc8d30swzDJLajtwO1EZ7DrAXhmkuD9ZSZz08jGjW8DHF9bbgfNX06sZFVWjOO1GAtpE/E1p3t8lYpNE1M0bpYoJZY2uwufGwvAOu2WZ1hVUdw3Cci1wBByIvvChileHYzocDTca+XmkvSz9Ll++Y7Vq3OuSf7yWfO7C1x35d61xKjhY6NlvDdW84U1zK/jp+iiQ7FShKLUcUIiJAfXahSoWU2Fmmp8BecRdiINyBcZWtfb/AMq9dFAaLk7SACeoXt5lN6iWhqHaz2lUqXaz2lFcQCIiBEIsZ0jSOMkkfHHxr4omwMD55XsvjObTZgwvubDok3A1245m4RLFLJJFihbLHUMayoh40N4t4IaMTDiAvnrJxc0hVcdG9jTyWpkze/ta/Nt26bdDOREVpmChSiALM2v5DyCpVU2v5DyCszTNja+R5wsYwve42FmgXOtPYjBJNzaW/wDZcRcVS8N5ZGzSihe6CM5yRF2FurpyOBAyOuw2ZZro9Dabgqx+U7nhl5In3a9uwnc4ZjMX1i9r2VMMRTm7Jm7FcE4zDRc6kNFtaakl22bt2vTrNksGv6Y+X9VnLBrmEuFgTkNQPWiv9vzrONifx95ioqsDvZPcU4t3snuKzWZzylYr4pOUB7XyxAUxYXwvDCX8Y04CdZFrn5LM4t3snuKYHeye4pONy7D4idCWem7OzV+0wZo5TNG50tRM1scwvNIHhuLDa269j3LMVXFu9k9xTi3eye4oUbDxGIqYhxdR3aVr79ZS1/2t2Jc970oquLd7J7inFu9k9xTsygpXScFuhJ8TfJc7xbvZPcV0fBdpDJLgjnN1i2xSgnc2YD867H6G5REVp3grc0DH9JoPaM+9XEQO7Ww1s2h2noOLeo84LVVUBjcWGxNtmqxXULntLn853Y3yCpqwildI1YerKUrNmsbRxtuWNEZJuTGMFzvNsj81ZrdEwTi08Ucu5zmASD/UM+6yzkVBrTtqjx70iaGZRyxNiBEUsbnWc4uIe02cATnaxb3rjnle7cKODcWkGMbI57HRucY3x4bjEACCCMxkN2pc7UcBaOg0ZWTVVqiVpc+nlGOFzSQ1kbLBx/WSTrGa3UKscqjznPxUKk6jqSfxI8pRXMIUFi1WMNyhFUWJgKBn1wiIshrCIoDs7bbX1IA07tZ7SiO1ntKhXkCVClECMG4aGNMkcckT6ji+UuLKaognJc6PGOi8GwuLnmXsQcqbAtdGHxSPkbTQubTOdJT01LA4kM4zW+Q3IzsecMrNJOeQoAtkMhuGpUuis17/ADYbVi2o2trv89lt+u3bz20KkRFcYgiKEAWptfyHkFr9OaO5VTS0+LBxgFn2uA5puLjaNn/C2E2v5DyCpQ4qUbMxwqzo1lUg7NO67bnm02jK3R8T6aOnZO2ZzHcojbO9129FmRAsCCcLmm5sdgI2/A/QzqSQS1BlM9RE64czmR3e3mvcTk52Vm7L55rsbKVmhhIxle/Ytx18Rw9WrUXScUnL7pLbLS3OnZbLpbrKybuUybPh+6hJNnw/dUcKfgXb+mYOD/yvs/aKVRWNnaITHHjEk7GOzAIY64xDsNvldVLW1DZopp5I4XTtngiDcD2NDJYw5oD8ZFmkOBxC9rLi0oRk2n8+fNz7tO8nt8Xa/fdW391uc20rC0lrhYjWFQseggMUUMTiC6OGJji3olzWgG3Ve6vqElFN22EG2m1e5kM1DtPmpUM1D5oqKiWbwMFRvMzF0ppOKkiM07nYS9rGhjQ6RzyRkASBkLnXqadap0TpeKrEjouMY+F4ZPDK0NfG4gX1axe7dhuw5K3wg0VyyFjGy8TNFM2SCTOzTcBwNhfee0BW9AaJkp3VFRUSieqqSBI5l8DWjCcIuM+diGWVg1alTocTfS9u+/j+tVrc2RjhuSuTk+M3Xe/da1rX1vdNbDarZaJ6Lu0eS1q2Wiei74m+SfBytiF2P0OfJ6GciIvREAiIgCVzmlD+c/tHkF0a19bowSOLw7C467i4P2UKkXJaFtCajK7NGiyptGyt/TiG9mfhrWKRsWZprab1JS1TAF8t64r00aSwQ0tG05yPM0gHsRjC0HtcSf8AQu7o2XcOrNeIekbSnKdJVDgbsiIp4z1R3Dv3y8/NasLHW5kxk7RscyiItxzQiIgD61REWM2hSoUoAxHUTNZcRmN2smw8U5A3e7w+yylKlmYWRi8gbvd4fZRyBu93h9lmIjMxWMXkDd7vD7JyBu93h9llIjMxGIaFg/UczYZjX3KeQN3u8PssktBtcXsbjqKlGZgYvIG73eH2TkDd7vD7LKRGZgYTtGsJvd2zd9k9Vs9p3h9llTShjS52QGtWmV0RFw8dHF1gWvc7h/VO8it06d9UWXaMjGtzhmBmW6zqGpT6rZ7R8Psr/K4j+pptY77bQezrTlbLhuLMtDhkbEE2Ft6LyFxVPcix6qZ7R8PspOi2H9TtVtn2V4VkerG3MgDPadXfsWSq6kFUVpq6JwSg7x0Nf6pZ7TvD7J6pZ7TvD7LYIquS0eii3jZ7zXHRkYtd7szYZtzNr5Zbge5T6pZ7TvD7LPIGWQyzHVsyUo5LR6KDjZ7zCGjGWAxHL+9yn1az2j/fyWaiTwlB/wCKIOTbuzC9Ws3n+/knq1ntH+/ks1EuR0OgvneK5gmgZcDE65BIFxcgWvs6x3q/TU4jBAJNzfYrthr2jUdoUqUMNSg80YpMQREVwBSiJiCIiAColha/pNDu0Z96rCIC9tUee6V4WU9FJpSJ80bZoLGlgIc17gYI3tGI5PJe52rZZeeeiyt0fFUzO0jxeJ0Q4iSpaHxB9yZL3uA4i1ievfng+lr/ABmu7YP5Ea5MOKujBKNkRlUbkm+Y3vDGalfXVL6IBtMXt4oNbhZfC3GWt2NLsRAWmVAepDwrEVMqRAUTEfWqIixm0IiIAlERABSoUpiYREQAREQAUhQpQBDmg5EX7Vb5LHqwNtuwi1rWV1ECLL6WMggtGYIyFjY5a/mpdTsNrsbkABlqA1AK6iLisWm0zBqY0Zg5Aaxq8yr6IgAiIgApUBSgAiIgQUKVCQBERABEUoBhERMQRECAJUIqkCPmr0tf4zXdsH8iNckut9LX+M13bB/IjXJLQtiKwiImIKbqEQB9dIiLKayUREAEREwClQiQMlERMQRQiQFSIiYgiIgAiIgCpERAgoREASiIgAiIgQRESAIiIAKURAmEREwAUqUQIpVSIgD5q9LX+M13bB/IjXJIi0LYisIiJiCIiAP/2Q==)

## Business Problem

FLO, an online shoe store, wants to segment its customers and determine marketing strategies based on these segments.

## Dataset Story

The dataset consists of information from customers who made their last purchases as OmniChannel (both online and offline) between the years 2020 and 2021.

### Variables
- master_id -- Unique customer number
- order_channel -- The channel used for shopping (Android, iOS, Desktop, Mobile)
- last_order_channel -- The channel used for the last purchase
- first_order_date -- The date of the customer's first purchase
- last_order_date -- The date of the customer's last purchase
- last_order_date_online -- The date of the customer's last online purchase
- last_order_date_offline -- The date of the customer's last offline purchase
- order_num_total_ever_online -- Total number of purchases made by the customer online
- order_num_total_ever_offline -- Total number of purchases made by the customer offline
- customer_value_total_ever_offline -- Total amount paid by the customer for offline purchases
- customer_value_total_ever_online -- Total amount paid by the customer for online purchases
